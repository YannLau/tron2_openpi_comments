"""serve_policy.py —— 策略（policy）推理服务端启动脚本。

这个脚本做什么？
-------------------------------------------------------------------------
它把一个训练好的动作模型（pi0.5 / TRON2 策略）加载成统一的 Policy 对象，
然后通过 WebSocket 对外提供推理服务。客户端（机器人程序）只需把当前观测
（相机图像 + 关节状态 + 可选任务指令）发过来，服务器就会返回一段预测的
动作序列（action chunk）。

快速上手（Quick Start）
-------------------------------------------------------------------------
1) 使用仓库自带的部署 YAML（推荐，TRON2 真机场景）：

       uv run scripts/serve_policy.py --profile configs/deploy/candy_server.yaml

   --profile 指向部署配置文件；其中 policy 小节描述加载哪个训练配置
   （policy.config）、哪个权重目录（policy.checkpoint_dir）等。YAML 的
   读取与解析逻辑见 src/openpi/shared/deploy_config.py。

2) 不指定配置文件，使用 --env 对应的“内置默认 checkpoint”
   （OpenPI 官方权重 / 示例权重，适用于快速试跑）：

       uv run scripts/serve_policy.py --env=libero

   --env 只在“没有通过 YAML 或 --policy 显式指定 checkpoint”时生效，
   具体取值见 DEFAULT_CHECKPOINT。

3) 直接显式指定训练配置与 checkpoint（OpenPI 文档中的经典写法）：

       uv run scripts/serve_policy.py policy:checkpoint \
           --policy.config=pi0_fast_droid \
           --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

   本脚本用 tyro 从 Args 数据类自动生成命令行参数，所以会出现
   `policy:checkpoint`（选择联合类型的哪一种）、`--policy.config`、
   `--policy.dir`（嵌套字段）这类写法。运行 `--help` 可查看完整参数。

服务启动后默认监听 0.0.0.0:8000（可用 --host / --port 或 YAML 覆盖），
机器人与它不在同一台机器时，把客户端配置文件里的 policy_host 改成这台
服务器的局域网 IP 即可。客户端示例见 examples/tron2/ 与
packages/openpi-client/。

main() 的执行流程（新手按这个顺序阅读即可）
-------------------------------------------------------------------------
  1. 解析并加载部署 YAML，取出 server / policy 两个配置小节。
  2. 按优先级创建 policy：
     命令行 Checkpoint > YAML 的 config + checkpoint_dir > --env 内置默认值。
  3. 把 action_horizon / rtc_enabled 等信息写入 metadata；
     客户端建立连接后，服务器会先发送这份 metadata。
  4. 预热（warmup）：用假观测调用一次推理，提前触发 JAX JIT 编译，
     避免第一个真实请求卡在漫长的编译上。
  5. 按需开启录制（record），保存每次推理的输入/输出以便调试。
  6. 启动 WebSocket 服务器，然后阻塞等待客户端连接。
"""

# ============================================================================
# 导入区 / Imports
# ============================================================================

# --- Python 标准库 ---
import dataclasses  # 数据类：用简洁语法定义“装参数的盒子”（Args 等）
import enum  # 枚举：给一组固定取值起名字（如环境类型）
import inspect  # 运行时检查函数签名，用来判断模型是否支持 RTC 参数
import logging  # 日志输出
import os  # 读取环境变量（如 OPENPI_JAX_CACHE_DIR）
import socket  # 获取本机主机名 / IP，用于启动时的提示日志
import time  # 计时（统计预热耗时）

# --- 第三方库 ---
import jax  # JAX：模型计算与编译框架（首次调用会触发 JIT 编译）
import numpy as np  # NumPy：数值数组（构造假观测等）
import tyro  # 类型驱动的命令行解析器：根据 Args 的注解自动生成命令行参数

# --- 本仓库内部模块 ---
# 惯例：import ... as _xxx 表示这个模块只在本脚本内部使用；
# 带下划线的前缀能避免与本地变量/函数重名，也标明“这是内部依赖”。
from openpi.policies import policy as _policy  # Policy 本体、PolicyRecorder（录制）
from openpi.policies import policy_config as _policy_config  # create_trained_policy()
from openpi.serving import websocket_policy_server  # 真正“跑起来”的 WebSocket 服务器
from openpi.shared import deploy_config as _deploy_config  # YAML 部署配置的读取工具
from openpi.training import config as _config  # 训练配置注册表（get_config）


# ============================================================================
# JAX 编译缓存
# ============================================================================
# pi0.5 这类模型很大，第一次推理会触发 JAX 的 JIT 编译，可能要等几分钟。
# 把编译结果持久化到磁盘后，下一次启动服务可以直接加载缓存，跳过重复编译。
#
# 缓存目录由环境变量 OPENPI_JAX_CACHE_DIR 控制：
#   * 不设置      -> 使用默认目录 /tmp/openpi_jax_cache
#   * 设置为空串  -> 关闭持久缓存（这里 if 判断会走到 False）
_JAX_CACHE_DIR = os.environ.get("OPENPI_JAX_CACHE_DIR", "/tmp/openpi_jax_cache")
if _JAX_CACHE_DIR:
    # 告诉 JAX 把编译产物写到哪个目录
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
    # 编译耗时超过 0 秒的算子都写入缓存（0 表示“几乎每次都缓存”）
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ============================================================================
# 环境枚举：--env 可选的取值
# ============================================================================
class EnvMode(enum.Enum):
    """支持的环境模式。

    每个成员对应一类机器人/仿真平台。只有当调用方“没有显式指定 checkpoint”
    （既没有 YAML 配置，也没有 --policy）时，--env 才会用来挑选默认权重。
    """

    ALOHA = "aloha"  # 真实 ALOHA 双臂机器人
    ALOHA_SIM = "aloha_sim"  # ALOHA 仿真环境
    DROID = "droid"  # DROID 机器人（OpenPI 远程推理文档的常用示例）
    LIBERO = "libero"  # LIBERO 仿真基准
    TRON2_REAL = "tron2_real"  # 本仓库的 TRON2 真机（默认值）


# ============================================================================
# 策略加载方式的两种选择（tyro 联合类型）
# ============================================================================
@dataclasses.dataclass
class Checkpoint:
    """通过命令行显式指定一个训练好的 checkpoint。

    tyro 会把这种“联合类型”渲染成选择子命令的形式，命令行写法：

        uv run scripts/serve_policy.py policy:checkpoint \
            --policy.config=pi05_tron2_example \
            --policy.dir=/path/to/checkpoints/exp/19999
    """

    # 训练配置名称，必须与 src/openpi/training/config.py 中注册的名字一致
    # （例如 "pi05_tron2_candy"、"pi0_aloha_sim"）。
    config: str

    # checkpoint 权重目录：
    #   * 本地路径（绝对或相对项目根目录）
    #   * 或 gs://... 云端对象存储路径（create_trained_policy 内部会自动下载）
    # 目录里通常有 params/（JAX 权重）或 model.safetensors（PyTorch 权重），
    # 以及 assets/（归一化统计量等配套数据）。
    dir: str


@dataclasses.dataclass
class Default:
    """“不指定 Checkpoint”时的占位类型：走默认策略。

    默认策略的来源依次为：
      1. YAML profile 中的 policy.config + policy.checkpoint_dir；
      2. --env 对应的内置 DEFAULT_CHECKPOINT。
    日常使用时不需要（也很难）手动构造它，Args.policy 默认就是 Default()。
    """


@dataclasses.dataclass
class Args:
    """serve_policy.py 的全部命令行参数。

    tyro 会根据这个数据类的字段名、类型和默认值自动生成命令行解析逻辑，
    所以这里每个字段本质上就是一个命令行选项。字段名里的下划线在命令行
    中通常会被 tyro 自动转换成连字符，例如 default_prompt 对应
    --default-prompt。运行 `uv run scripts/serve_policy.py --help` 可查看
    自动生成的完整帮助。
    """

    # 首选入口：指定部署 YAML 文件（TRON2 真机推荐用法）。
    # 例如：--profile configs/deploy/candy_server.yaml
    profile: str | None = None

    # --deploy-config 是 --profile 的旧别名，仅为兼容旧命令保留；
    # 两者不能同时传入（会抛 ValueError，见 deploy_config.select_profile_path）。
    deploy_config: str | None = None

    # 服务的机器人环境。只在“没有通过 YAML 或 --policy 指定 checkpoint”
    # 时用于查找 DEFAULT_CHECKPOINT（见 create_default_policy）。
    env: EnvMode = EnvMode.TRON2_REAL

    # 当客户端传来的观测里没有任务指令（prompt）时使用的默认指令文本。
    # 也可以在 YAML 的 policy.default_prompt 里配置；命令行优先级更高：
    # 只有这里为 None 时才会去读 YAML。
    default_prompt: str | None = None

    # 服务器监听地址与端口。优先级：命令行 > YAML server 小节 > 内置默认值
    # （0.0.0.0:8000）。0.0.0.0 表示监听所有网卡，允许局域网内的客户端连入。
    host: str | None = None
    port: int | None = None

    # 是否把每次推理的输入/输出原样保存到 policy_records/ 目录（调试用）。
    # 也可在 YAML 的 policy.record 里开启；命令行与 YAML 任一为 true 即生效。
    record: bool = False

    # “如何加载策略”：
    #   policy:checkpoint + --policy.config / --policy.dir  -> 显式加载权重；
    #   不指定（默认 Default()）                            -> 走 YAML/内置默认。
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ---------------- 历史遗留参数（已废弃，仅为兼容保留） ----------------
    # RTC 支持现在由脚本自动探测（见 _policy_supports_rtc），RTC 运行参数也
    # 由客户端在推理请求中携带。rtc_enabled 仅用于“模型不支持 RTC 却要求
    # 开启”时打一条告警；其余三个字段已完全不再使用。
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "exp"


# ============================================================================
# 内置默认 checkpoint 表：--env 对应的官方/示例权重
# ============================================================================
# 只在“既没有 YAML policy.config/checkpoint_dir，也没有 --policy”时使用。
# 字典的 key 是环境，value 是一个 Checkpoint（config 名称 + 权重目录）。
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.TRON2_REAL: Checkpoint(
        config="pi05_tron2_example",
        dir="checkpoints/pi05_tron2_example/example_checkpoint/19999",
    ),
    # 注意：TRON2 真机这里只放了“示例”配置，仓库不携带真实权重；
    # 实际部署请用 --profile 指向你自己的权重目录。
}


# ============================================================================
# 训练配置（TrainConfig）的读取与部署期覆盖
# ============================================================================
# 每个 checkpoint 都对应一套在 src/openpi/training/config.py 里注册好的
# TrainConfig（模型结构、数据格式、变换管线等）。部署 YAML 允许少量字段
# 覆盖训练时的取值（例如换 repo_id、调整 action_horizon）。
# 下面几个 _with_* 小函数就是“逐个覆盖字段”的工具；它们都遵循同一约定：
#   * 参数为 None  -> 表示 YAML 没写这项，原样返回，不做任何修改；
#   * 参数有值     -> 覆盖对应字段并返回一份新的 TrainConfig。


def _with_repo_id(train_config: _config.TrainConfig, repo_id: str | None) -> _config.TrainConfig:
    """覆盖训练配置的数据/仓库 ID（data.repo_id）。

    repo_id 用来在 checkpoint 目录里定位配套的 assets（例如归一化统计量
    norm_stats.json），必须与 checkpoint 训练时使用的 ID 保持一致。
    """
    if repo_id is None:
        return train_config
    # dataclasses.replace 生成一份“修改后的新对象”，不会改动传入的旧对象。
    # 这里先替换内层 data 字段，再把它写回外层 train_config。
    return dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, repo_id=repo_id))


def _with_action_horizon(
    train_config: _config.TrainConfig,
    action_horizon: int | None,
) -> _config.TrainConfig:
    """覆盖模型每次推理输出的动作步数（action_horizon）。

    一次推理会预测未来 action_horizon 步的动作，客户端按这个长度连续播放。
    覆盖值需要和权重本身匹配（不同 checkpoint 训练时的动作长度不同）。
    """
    if action_horizon is None:
        return train_config
    # YAML 读进来的可能是字符串（如 "30"），统一转成 int 再做校验
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"policy.action_horizon must be positive, got {action_horizon}")
    # 只有模型配置本身带 action_horizon 字段时才能覆盖，否则抛错说明白
    if not hasattr(train_config.model, "action_horizon"):
        raise ValueError(f"Model config {type(train_config.model).__name__} does not support action_horizon")

    # 先替换模型配置中的动作长度，再把模型配置写回 TrainConfig
    model_config = dataclasses.replace(train_config.model, action_horizon=action_horizon)
    logging.info("Overriding inference action_horizon to %d from deploy config", action_horizon)
    return dataclasses.replace(train_config, model=model_config)


def _with_state_dim(train_config: _config.TrainConfig, state_dim: int | None) -> _config.TrainConfig:
    """覆盖本体状态向量的维度（state_dim）。

    对 TRON2 来说，state_dim 是发往策略的关节状态长度：16 通常表示
    “双臂 + 夹爪”，18 会额外包含头部关节。它同时会写进 policy_metadata，
    随 metadata 一起发给客户端。
    """
    if state_dim is None:
        return train_config
    state_dim = int(state_dim)
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    # 非 TRON2 的很多训练配置没有 state_dim 概念；不强制报错，打警告忽略即可
    if not hasattr(train_config.data, "state_dim"):
        logging.warning("Config %s does not support state_dim; ignoring override.", train_config.name)
        return train_config

    # 覆盖数据配置里的 state_dim，同时记录到 policy_metadata（供客户端读取）
    data_config = dataclasses.replace(train_config.data, state_dim=state_dim)
    policy_metadata = dict(train_config.policy_metadata or {})
    policy_metadata["state_dim"] = state_dim
    logging.info("Overriding TRON2 state/action output dim to %d from deploy config", state_dim)
    return dataclasses.replace(train_config, data=data_config, policy_metadata=policy_metadata)


def _with_delta_actions(train_config: _config.TrainConfig, use_delta: bool | None) -> _config.TrainConfig:
    """覆盖动作表示方式：绝对关节角 vs 关节角增量（delta）。

    若权重训练时用的是“增量动作”，客户端/控制侧通常也要按增量语义执行；
    该项必须与训练配置匹配，因此默认不写时保持训练配置原样。
    """
    if use_delta is None:
        return train_config
    if not hasattr(train_config.data, "use_delta_joint_actions"):
        logging.warning("Config %s does not support use_delta_joint_actions; ignoring override.", train_config.name)
        return train_config
    return dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, use_delta_joint_actions=use_delta),
    )


def _get_train_config(
    config_name: str,
    *,
    repo_id: str | None = None,
    action_horizon: int | None = None,
    state_dim: int | None = None,
    use_delta_joint_actions: bool | None = None,
) -> _config.TrainConfig:
    """按名称取出训练配置，并依次套用所有部署期覆盖项。

    * 后面的参数都是 keyword-only：调用时必须写成 repo_id=... 的形式，
      防止传参顺序出错。每个覆盖项默认 None = 不覆盖。
    """
    # 1) 从注册表按名称取出基础训练配置
    train_config = _config.get_config(config_name)
    # 2) 依次覆盖；每步都基于上一步返回的“新配置”，形成一条处理流水线
    train_config = _with_repo_id(train_config, repo_id)
    train_config = _with_action_horizon(train_config, action_horizon)
    train_config = _with_state_dim(train_config, state_dim)
    train_config = _with_delta_actions(train_config, use_delta_joint_actions)
    return train_config


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """按环境创建内置默认策略（只用于 --env 快速试跑）。

    流程：env -> 查 DEFAULT_CHECKPOINT -> 得到 Checkpoint(config, dir)
    -> 调用 create_trained_policy() 加载权重并组装成 Policy。
    """
    # 海象运算符 := 同时完成“查表”和“判断查到没有”：
    # 查到就把 checkpoint 赋给本地变量；查不到（None）则不进入 if。
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        # create_trained_policy() 会负责：必要时下载权重、读取归一化统计量、
        # 按训练配置组装输入/输出变换，最后返回一个可直接 infer() 的 Policy。
        return _policy_config.create_trained_policy(
            _get_train_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _policy_bool(policy_profile: dict, key: str) -> bool | None:
    """从 YAML 的 policy 小节安全读取一个布尔字段。

    YAML 里 true/false/yes/no/on/off/"1"/"0" 等写法都可能出现，
    统一交给 deploy_config.bool_value 解析成标准 Python bool。
    字段不存在时返回 None，表示“不覆盖，使用训练配置里的默认值”。
    """
    if key not in policy_profile:
        return None
    return _deploy_config.bool_value(policy_profile[key])


def create_policy(args: Args, config_profile: dict) -> _policy.Policy:
    """把命令行参数 + YAML 配置合成最终要服务的 Policy。

    创建优先级（从高到低）：
      1. args.policy 为 Checkpoint()：命令行显式给出 config + dir；
      2. args.policy 为 Default()，且 YAML 的 policy 小节写了
         config + checkpoint_dir：使用 YAML 指定的权重；
      3. 其余情况：使用 args.env 对应的内置默认权重（DEFAULT_CHECKPOINT）。

    注意：无论走哪条路，YAML 里的 repo_id / action_horizon / state_dim /
    use_delta_joint_actions 都会作为“部署期覆盖项”传给训练配置。
    """
    # section() 从整个 YAML 字典里取出一个子块；子块缺失时返回 {}，
    # 因此后面用 .get(...) 拿默认值不会因为 KeyError 崩溃。
    policy_profile = _deploy_config.section(config_profile, "policy")
    client_profile = _deploy_config.section(config_profile, "client")

    # 默认任务指令：命令行优先级更高，只有命令行没给才读 YAML
    default_prompt = args.default_prompt
    if default_prompt is None:
        default_prompt = policy_profile.get("default_prompt")

    # 从 YAML 收集“部署期覆盖项”
    repo_id = policy_profile.get("repo_id")
    action_horizon = policy_profile.get("action_horizon")
    # policy.state_dim 没写时回退到 client.state_dim（两个部署模板里通常一致）
    state_dim = policy_profile.get("state_dim", client_profile.get("state_dim"))
    use_delta_joint_actions = _policy_bool(policy_profile, "use_delta_joint_actions")

    # match/case 是 Python 3.10+ 的“结构化模式匹配”，
    # 这里用来区分 args.policy 到底是联合类型里的哪一种。
    match args.policy:
        case Checkpoint():
            # ---- 情况 1：命令行显式指定（policy:checkpoint ...）----
            train_config = _get_train_config(
                args.policy.config,
                repo_id=repo_id,
                action_horizon=action_horizon,
                state_dim=state_dim,
                use_delta_joint_actions=use_delta_joint_actions,
            )
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=default_prompt,
            )
        case Default():
            # ---- 情况 2：先检查 YAML 是否指定了权重 ----
            # 兼容 checkpoint_dir 与更旧的 dir 两种字段写法
            config_name = policy_profile.get("config")
            checkpoint_dir = policy_profile.get("checkpoint_dir") or policy_profile.get("dir")
            if config_name or checkpoint_dir:
                # config 与 checkpoint_dir 必须成对出现，避免只写一半造成困惑
                if not config_name or not checkpoint_dir:
                    raise ValueError("YAML policy config requires both policy.config and policy.checkpoint_dir")
                train_config = _get_train_config(
                    str(config_name),
                    repo_id=repo_id,
                    action_horizon=action_horizon,
                    state_dim=state_dim,
                    use_delta_joint_actions=use_delta_joint_actions,
                )
                return _policy_config.create_trained_policy(
                    train_config,
                    str(checkpoint_dir),
                    default_prompt=default_prompt,
                )
            # ---- 情况 3：YAML 也没写 -> 落到 --env 的内置默认权重 ----
            return create_default_policy(args.env, default_prompt=default_prompt)


# ============================================================================
# RTC 能力探测与模型预热
# ============================================================================
# RTC（Real-Time Chunking，实时动作分块）是 TRON2 真机客户端使用的一种部署
# 模式：推理可以带上“上一段动作的残块、已等待帧数”等上下文，让新旧动作
# 平滑衔接。不是所有模型都实现了这套参数，因此脚本在启动时动态探测。


def _policy_supports_rtc(policy: _policy.Policy) -> bool:
    """判断当前加载的模型是否支持 RTC 推理参数。

    原理：RTC 模式要求模型 sample_actions() 能接收 inference_delay、
    prev_chunk_left_over 等参数。不同模型/不同训练配置的实现不同，
    这里通过 inspect 检查函数签名来“动态探测”，而不是硬编码一份模型名单。
    """
    # Policy 把底层模型存在“私有属性” _model 上；用 getattr + 默认 None，
    # 保证即使模型内部结构发生变化，这里也不会抛 AttributeError。
    model = getattr(policy, "_model", None)
    sample_actions = getattr(model, "sample_actions", None)
    if sample_actions is None:
        # 拿不到 sample_actions 方法 -> 无法判断，按不支持处理
        return False
    # inspect.signature 返回函数签名，.parameters 是所有形参名
    params = inspect.signature(sample_actions).parameters
    # <= 是“子集判断”：右边参数名集合必须完整包含左边这 5 个 RTC 参数，
    # 缺任何一个都视为不支持 RTC。
    return {
        "inference_delay",
        "prev_chunk_left_over",
        "prev_chunk_left_over_len",
        "prefix_horizon",
        "max_guidance_weight",
    } <= set(params)


def warmup_policy(policy: _policy.Policy, rtc_supported: bool) -> None:
    """在收到第一个真实客户端请求前，用假观测触发推理、完成 JIT 编译。

    为什么要预热？JAX 在第一次调用 infer() 时才会把模型编译成可执行程序，
    这个编译过程可能耗时几分钟。服务端提前自己跑一遍，客户端连接后的第一个
    请求就是“热”的，不会卡在编译上。

    预热覆盖两条路径：
      1. 普通推理路径（任何模型都需要编译）；
      2. trained-RTC 推理路径（只有 rtc_supported=True 才编译）。

    注意：假观测只是“跑通流程、触发编译”用，数值没有意义。若某个自定义
    任务模型的输入结构不同，预热可能失败；这里用 try/except 包住，失败只
    记录错误日志，不影响服务器继续启动。
    """
    # getattr(obj, name, 默认值)：读取不到属性时返回默认值。
    # 这样即使模型没有暴露对应属性，预热也不会因为 AttributeError 中断。
    model = getattr(policy, "_model", None)
    action_horizon = int(getattr(model, "action_horizon", 50))  # 动作块长度（步数）
    action_dim = int(getattr(model, "action_dim", 32))  # 每个动作向量的维度
    state_dim = int(policy.metadata.get("state_dim", 16))  # 关节状态向量维度

    # 构造一份与真实观测结构一致的“假观测”：
    #   state  : 一维关节状态（全 0）
    #   images : 三路相机（高位相机 + 左右手腕相机），形状 (C, H, W)
    #            = (3, 224, 224)：3 是 RGB 通道数，224 是模型输入分辨率。
    # 数值故意全填 0——预热只关心能否走通整条计算图并完成编译。
    dummy_obs = {
        "state": np.zeros(state_dim, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
    }

    # ---- 第 1 次预热：普通（非 RTC）推理路径 ----
    logging.info("[warmup] Compiling standard inference path...")
    t0 = time.monotonic()  # time.monotonic() 是单调时钟，适合测量耗时
    try:
        policy.infer(dummy_obs)
        logging.info("[warmup] Standard inference path complete in %.1fs", time.monotonic() - t0)
    except Exception:
        # logging.exception() 会在 ERROR 日志里自动带上当前异常堆栈
        logging.exception("[warmup] Standard inference path failed; continuing server startup")

    # ---- 第 2 次预热（仅 RTC 模型）：trained-RTC 推理路径 ----
    if rtc_supported:
        # prev_chunk_left_over：上一次预测中“还没执行完的动作残块”，
        # 形状为 (action_horizon, action_dim)，这里用全 0 占位。
        dummy_chunk = np.zeros((action_horizon, action_dim), dtype=np.float32)
        logging.info("[warmup] Compiling trained-RTC inference path...")
        t0 = time.monotonic()
        try:
            policy.infer(
                dummy_obs,
                inference_delay=10,  # 模拟 RTC 请求中的初始等待/延迟参数
                prev_chunk_left_over=dummy_chunk,  # 上一段动作占位
                trained_rtc_mode=True,  # 走训练时 RTC 条件化路径
            )
            logging.info("[warmup] trained-RTC path complete in %.1fs", time.monotonic() - t0)
        except Exception:
            logging.exception("[warmup] trained-RTC path failed; continuing server startup")


def main(args: Args) -> None:
    """主流程：加载配置 -> 创建 policy -> 补充 metadata -> 预热 -> 启动服务。"""
    # 1) 选择部署配置文件路径（--profile 与旧参数 --deploy-config 二选一）
    profile_path = _deploy_config.select_profile_path(args.profile, args.deploy_config)
    # 2) 把 YAML 读成 Python 字典；没有传文件时返回 {}，此时全部使用默认值
    config_profile = _deploy_config.load_deploy_config(profile_path)
    # 3) 取出 server / policy 两个子配置块（配置块缺失时返回空 dict）
    server_profile = _deploy_config.section(config_profile, "server")
    policy_profile = _deploy_config.section(config_profile, "policy")

    # 4) 创建 policy（优先级：命令行 Checkpoint > YAML > --env 内置默认）
    policy = create_policy(args, config_profile)
    # metadata 是一个可写 dict；客户端连接后，服务器会在握手阶段先发它
    policy_metadata = policy.metadata

    # 把模型的动作块长度 action_horizon 也写进 metadata，
    # 供客户端读取“H”（一次推理预测多少步动作）
    model = getattr(policy, "_model", None)
    action_horizon = getattr(model, "action_horizon", None)
    if action_horizon is not None:
        policy_metadata["action_horizon"] = int(action_horizon)

    # 探测模型是否支持 RTC，并把结果发布到 metadata（客户端据此选协议）
    rtc_supported = _policy_supports_rtc(policy)
    policy_metadata["rtc_enabled"] = rtc_supported
    if rtc_supported:
        logging.info("RTC supported; client supplies execution_horizon, delay, and guidance weight.")
    else:
        logging.info("RTC not supported by this model.")
    # 兼容旧命令/旧 YAML：用户显式要求了 RTC，但模型不支持时只告警，不中断
    if (args.rtc_enabled or _deploy_config.bool_value(server_profile.get("rtc_enabled", False))) and not rtc_supported:
        logging.warning("RTC was requested, but this model does not expose RTC parameters.")

    # 5) 预热：提前触发 JIT 编译，避免第一个真实客户端请求等太久
    warmup_policy(policy, rtc_supported)

    # 6) 可选录制：用 PolicyRecorder 把 policy 包一层，
    #    之后每次 infer 的输入/输出都会保存到 policy_records/ 目录
    #    （实现见 src/openpi/policies/policy.py 中的 PolicyRecorder）
    record = args.record or _deploy_config.bool_value(policy_profile.get("record", False))
    if record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 7) 打印本机主机名与 IP，方便填写客户端配置里的 policy_host
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 8) 确定监听地址与端口：命令行 > YAML server 小节 > 默认 0.0.0.0:8000
    host = args.host or str(server_profile.get("host", "0.0.0.0"))
    port = args.port if args.port is not None else int(server_profile.get("port", 8000))

    # 9) 构造 WebSocket 策略服务器并开始服务。
    #    协议简述：客户端连上后先收到 metadata，然后反复“发观测 -> 收动作”。
    #    服务端实现见 src/openpi/serving/websocket_policy_server.py。
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy_metadata,
    )
    # serve_forever() 会一直阻塞运行，直到进程被终止（例如 Ctrl+C）
    server.serve_forever()


if __name__ == "__main__":
    # 只有“直接运行本文件”（python scripts/serve_policy.py）时才执行到这里；
    # 被其他模块 import 时不会触发。
    # basicConfig 一次性配置根日志：输出 INFO 及以上级别；
    # force=True 表示覆盖之前可能已经存在的日志配置。
    logging.basicConfig(level=logging.INFO, force=True)
    # tyro.cli(Args)：根据 Args 数据类的字段/类型自动解析命令行参数，
    # 构造好 Args 实例后交给 main() 执行。
    main(tyro.cli(Args))
