"""openpi 的训练/部署“配置中心”——在这里选择或自定义一个 TrainConfig。

这个文件是干什么的？
-------------------------------------------------------------------------
openpi 把“用什么模型、在什么数据上训练、怎么变换数据、训练多久”打包成一个
TrainConfig 对象。本文件负责：

  1. 定义各种配置数据类型（AssetsConfig、DataConfig、TrainConfig 等）；
  2. 提供若干“数据配置工厂”，把数据集原始格式转换成模型需要的格式；
  3. 在 _CONFIGS 中注册一批开箱即用的命名配置（如 pi0_aloha_sim、
     pi05_tron2_candy、debug），并通过 get_config()/cli() 按名字取用。

换句话说：训练脚本从这里拿配置，推理服务（scripts/serve_policy.py 中的
policy.config）也从这里按名字引用同一种配置。部署 YAML 里出现的
action_horizon / state_dim / use_delta_joint_actions 等覆盖项，最终也会
作用到本文件生成的 TrainConfig 上。

快速上手 / Quick Start
-------------------------------------------------------------------------
1) 查看当前注册了哪些配置（每个名字都是一个可选的子命令）：

       uv run scripts/train.py --help

2) 快速试跑“调试配置”（假数据、10 步、立即覆盖旧 checkpoint）：

       uv run scripts/train.py debug

3) 正式训练：子命令名 = 配置名，后面可以按字段覆盖超参数，
   exp-name 会被拼进 checkpoint 目录，所以每次运行都应取不同名字：

       uv run scripts/train.py pi0_aloha_sim --exp-name my_first_run

   JAX 与 PyTorch 两条训练入口都使用同一个 cli()：
     - scripts/train.py           （JAX 训练）
     - scripts/train_pytorch.py   （PyTorch/DDP 训练）

4) 在 Python 代码里按名字取配置：

       from openpi.training.config import get_config
       cfg = get_config("pi05_tron2_candy")

阅读地图（建议新手按这个顺序读）
-------------------------------------------------------------------------
  AssetsConfig                assets（归一化统计量等）放在哪里；
  DataConfig                  数据加载器最终使用的“数据配置结果”；
  GroupFactory / ModelTransformFactory
                              如何把“原始数据格式”翻译成“模型输入”；
  DataConfigFactory(ABC)      数据配置的“工厂基类”：输入一个数据集，
                              产出上面那个 DataConfig；
  FakeDataConfig...LeRobotDROIDDataConfig
                              针对不同数据集的具体工厂实现；
  TrainConfig                 一次训练任务的总配置（模型 + 数据 + 超参数）；
  _CONFIGS / _CONFIGS_DICT    命名配置注册表，cli() 与 get_config() 的数据源。

理解“数据变换流水线”是读懂本文件的关键
-------------------------------------------------------------------------
对于一条训练样本，数据大致按下面的顺序流动：

  数据集原始样本
    → repack_transforms    只重排 key 名，让数据集格式贴近推理环境
                            （只在训练/数据加载阶段使用）；
    → data_transforms      机器人特有的输入/输出变换（训练和推理都会用）；
    → 归一化               用 norm_stats 把状态/动作缩放到模型习惯的范围；
    → model_transforms     模型特有变换（如 resize 图像、把 prompt 分词、
                            补齐 state/action 到固定长度）；
    → 模型推理

推理输出还会做对称的“逆变换”（反归一化、恢复绝对动作等），
具体顺序由 policy_config.create_trained_policy() 统一组装。

想新增自己的任务时，通常不需要改本文件：TRON2 任务推荐复制
configs/train/tron2_tasks/example.yaml 后用 scripts/train_tron2_task.py
训练（YAML 会自动转成 TrainConfig）。只有当数据集格式特殊、需要新的变换
管线时，才在这里仿照 LeRobotLiberoDataConfig 等类新增一个工厂。
"""

# ============================================================================
# 导入区 / Imports
# ============================================================================

# --- Python 标准库 ---
import abc  # 抽象基类：定义 DataConfigFactory 的“接口契约”
from collections.abc import Sequence
import dataclasses  # 数据类：本文件几乎全部配置都是 dataclass
import difflib  # 字符串相似度匹配，用于 get_config() 的“你是不是想找…”提示
import logging  # 日志
import pathlib  # 路径操作（assets/checkpoint 目录）
from typing import Any, Literal, Protocol, TypeAlias

# --- 第三方库 ---
import etils.epath as epath  # “支持 gs:// 的路径库”，用于云端 assets
import flax.nnx as nnx  # Flax 新式 NN 模块库（模型参数结构）
from typing_extensions import override  # 更明确的“重写父类方法”标记
import tyro  # 类型驱动 CLI：把 TrainConfig 的字段自动变成命令行参数

# --- 本仓库内部模块 ---
# 注意别名风格：有的模块如 pi0_config 不带下划线是因为本文件会公开引用；
# 带 _ 前缀的别名（如 _model）表示“只是内部依赖，避免与类名/变量冲突”。
import openpi.models.model as _model  # 模型基类/模型类型枚举
import openpi.models.pi0_config as pi0_config  # pi0 / pi0.5 模型配置
import openpi.models.pi0_fast as pi0_fast  # pi0-FAST（快速扩散模型）配置
import openpi.models.tokenizer as _tokenizer  # prompt/action 分词器
import openpi.policies.aloha_policy as aloha_policy  # ALOHA 输入/输出变换
import openpi.policies.tron2_policy as tron2_policy  # TRON2 输入/输出变换
import openpi.policies.droid_policy as droid_policy  # DROID 输入/输出变换
import openpi.policies.libero_policy as libero_policy  # LIBERO 输入/输出变换
import openpi.shared.download as _download  # 自动下载 gs:// 远端资源
import openpi.shared.normalize as _normalize  # 归一化统计量读写
import openpi.training.droid_rlds_dataset as droid_rlds_dataset  # DROID RLDS 数据集
import openpi.training.misc.polaris_config as polaris_config  # PolaRiS 数据集配置
import openpi.training.misc.roboarena_config as roboarena_config  # RoboArena 数据集配置
import openpi.training.optimizer as _optimizer  # 学习率/优化器配置
import openpi.training.weight_loaders as weight_loaders  # 权重加载器（预训练初始化）
import openpi.transforms as _transforms  # 全部数据变换原语

# 模型类型（PI0 / PI05 / PI0_FAST）：用来区分三种模型家族的数据格式
ModelType: TypeAlias = _model.ModelType
# 参数过滤器的别名。单独起一个别名是为了绕开 tyro 直接使用
# nnx.filterlib.Filter 时的解析问题；它用于“冻结哪些参数”等场景。
Filter: TypeAlias = nnx.filterlib.Filter


# ============================================================================
# Assets（数据资产）配置
# ============================================================================
# 这里的 “assets” 指构建数据/推理管线所需的配套文件，最典型的是归一化
# 统计量 norm_stats.json：它记录每个状态/动作维度的 mean/std 等数值。
# 训练与推理必须使用同一份统计量，否则状态和动作会被错误地缩放。
@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """指定 assets 放在哪里。

    平时可以不设置：默认从“当前训练的 assets 目录”（TrainConfig.assets_dirs）
    里查找。微调（fine-tuning）时常用它指向“基座模型的 assets”，从而复用
    基座模型训练时使用的同一套归一化统计量。示例：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # assets 所在目录（本地路径或 gs:// 云端路径）。
    # 留空时使用训练配置自带的 assets 目录。
    assets_dir: str | None = None

    # assets 的标识名（对应目录下的子目录名）。留空时默认使用数据集 repo_id。
    # 例如 asset_id="trossen" 会去 {assets_dir}/trossen/ 找 norm_stats.json。
    asset_id: str | None = None


# ============================================================================
# DataConfig：数据加载器最终使用的“数据配置结果”
# ============================================================================
# DataConfig 是一份 frozen（不可变）配置，描述“这份数据要经过哪些变换、
# 归一化统计量是什么、动作序列从哪些 key 读取”。
#
# 新手注意区分：
#   - TrainConfig.data 的类型是 DataConfigFactory（工厂），负责按模型/目录
#     动态组装下面的 DataConfig；
#   - 训练或推理代码在运行时调用 factory.create(assets_dirs, model_config)
#     才会得到这里的 DataConfig。
@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot 数据集 repo id（Hugging Face id、本地目录等）；None 时生成假数据
    repo_id: str | None = None
    # assets 目录里的子目录 id（通常等于 repo_id）
    asset_id: str | None = None
    # 预计算好的归一化统计量。为 None 表示不做归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # repack 变换：把“数据集自己的字段名”翻译成“openpi 统一字段名”。
    # 例如把 observation.images.top 映射为 images.cam_high。
    # 只在训练/数据加载阶段生效，推理环境不需要（推理端字段本来就统一）。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 数据变换：通常包含机器人特有的变换（例如 ALOHA/TRON2/DROID 的坐标约定、
    # 夹爪开合定义）。训练与推理都会使用，且作用在归一化之前。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 模型变换：在归一化之后、进入模型之前使用（例如把 prompt 分词、
    # 把 state/action 补齐到模型要求长度）。输出侧还有对应的逆变换。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 归一化方式：True 用分位数（quantile）归一化；False 用普通 z-score 归一化。
    # pi0.5 / pi0-FAST 使用分位数归一化，原版 pi0 使用 z-score；
    # 工厂会自动按模型类型设置，无需手动改。
    use_quantile_norm: bool = False

    # 数据加载器从哪些 key 拼接“动作序列”。序列长度由模型配置里的
    # action_horizon 决定；若你的 LeRobot 数据集用别的 key 存动作，改这里。
    action_sequence_keys: Sequence[str] = ("actions",)

    # True 时直接用 LeRobot 数据集的 task 文本作为 prompt（任务指令）。
    prompt_from_task: bool = False

    # 以下三个字段只用于 RLDS 数据加载器（目前仅 DROID 全量数据使用）：
    #   rlds_data_dir —— RLDS 数据集根目录；
    #   action_space  —— DROID 动作空间（例如关节绝对位置）；
    #   datasets      —— 采样哪些数据集：名字、版本、权重，可选过滤文件。
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


# ============================================================================
# 变换工厂：根据模型类型生成 _transforms.Group
# ============================================================================
class GroupFactory(Protocol):
    """“变换工厂”的接口：给定模型配置，返回一组输入/输出变换。

    Protocol 在这里相当于一个抽象接口——任何实现了 __call__ 的对象
    （dataclass、lambda 函数等）都可以当作 GroupFactory 使用。
    例如下面 _CONFIGS 里就出现过 `data_transforms=lambda model: ...`。
    """

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """标准 pi0 / pi0.5 / pi0-FAST 模型的“模型变换工厂”。

    它按模型类型返回“归一化之后、进入模型之前”需要的输入变换：
      - InjectDefaultPrompt：观测里没有 prompt 时注入默认任务指令；
      - ResizeImages：把图像缩放到模型输入分辨率 224x224；
      - TokenizePrompt / TokenizeFASTInputs：把 prompt（以及部分模型里的
        状态、动作）编码成模型能处理的 token；
      - PadStatesAndActions：把 state/action 补齐到模型要求的固定长度。

    pi0-FAST 还需要在输出侧用 ExtractFASTActions 把离散 token 解码回连续
    动作，所以只有 PI0_FAST 分支带 outputs。
    """

    # 若观测数据里没有 prompt key，就用这里的文本作为默认任务指令
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # match/case（Python 3.10+ 结构化匹配）：按模型类型分三条流水线
        match model_config.model_type:
            case _model.ModelType.PI0:
                # pi0 原版：图像 + 文本 prompt + 连续 state/action
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # pi0.5：与 pi0 基本相同，只是支持把状态也作为离散 token 输入
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                # pi0-FAST：动作也被离散化（分箱）成 token，输出时再解码。
                # 分词器允许通过 fast_model_tokenizer(_kwargs) 自定义。
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


# ============================================================================
# DataConfigFactory：数据集 -> DataConfig 的工厂基类
# ============================================================================
# 为什么需要“工厂”？因为 DataConfig 的有些字段要运行时才能确定：
#   - 归一化统计量在 assets 目录里，可能需要按 asset_id 加载甚至下载；
#   - 变换内容依赖模型类型（PI0 / PI05 / PI0_FAST）。
# 因此 TrainConfig.data 里存放的是“工厂”，训练/推理代码随后调用
# data.create(assets_dirs, model_config) 生成真正的 DataConfig。
# 要接入一种新数据集，通常就是继承本类并实现 create()。
@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # 数据集 repo id（LeRobot Hugging Face id / 本地路径）。
    # tyro.MISSING 表示“用户没有显式填写”，它和 None 不同：MISSING 会在此处
    # 被解析成 None（即“不加载真实数据集”），同时允许子类覆盖默认值。
    repo_id: str = tyro.MISSING
    # assets（归一化统计量等）从哪个目录/id 加载
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # “基底 DataConfig”：工厂以它为基础，再用运行时算出的值覆盖字段。
    # tyro.conf.Suppress 表示不把该字段暴露成命令行参数（内部使用）。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """子类必须实现：结合 assets 目录与模型配置，生成真正的 DataConfig。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """完成所有工厂“共用的逻辑”：repo_id、asset_id、norm stats、归一化方式。

        子类 create() 的惯用写法是先调用本方法拿到基础配置，再用
        dataclasses.replace() 补上自己的 repack/data/model 变换。
        """
        # tyro.MISSING -> 转成 None（表示未显式指定数据集）
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # asset_id 优先级：AssetsConfig.asset_id 优先，其次用 repo_id
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            # 尝试加载归一化统计量；找不到时 _load_norm_stats 返回 None
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            # 原始 pi0 用普通 z-score；pi0.5/pi0-FAST 用分位数归一化
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        """从 assets 目录加载归一化统计量；不存在则跳过并返回 None。"""
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            # maybe_download：如果路径是 gs:// 会自动下载到本地缓存再读取
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


# ============================================================================
# 具体数据集工厂实现
# ============================================================================
# 下面的类都继承 DataConfigFactory，差别只在于“数据集长什么样、需要哪套
# repack/data/model 变换”。它们最终都会被某个 TrainConfig.data 引用。
@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """假数据工厂：不读真实数据集，供冒烟测试/调试（配置名如 debug）。"""

    # 固定 repo_id="fake"，数据加载器会据此生成随机样本
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 假数据不需要 assets 或真实变换：返回最小 DataConfig 即可
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """通用工厂：把传入的变换工厂实例化后组装进 DataConfig。

    适用于“数据集本身没有特殊 repack 逻辑、只需要注入变换”的场景
    （例如 _CONFIGS 里的 DROID 推理配置）。也可直接传 lambda：
        data_transforms=lambda model: _transforms.Group(...)
    """

    # 数据变换的工厂（运行时由 create() 调用，返回 _transforms.Group）
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型变换的工厂（默认使用上面通用的 ModelTransformFactory）
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 先取公共基础配置，再把两个变换工厂“跑一次”得到真正的变换
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """ALOHA 双臂机器人（LeRobot 数据集格式）的数据工厂。

    它把标准 ALOHA 数据整理成模型需要的格式：
      - repack：把数据集 key 映射到 openpi 通用 key；
      - data_transforms：套用 ALOHA 坐标/夹爪约定（AlohaInputs/Outputs）；
      - 可选：把绝对关节角转成“相对当前状态的增量”供模型学习。
    """

    # 若为 True，把关节角转成相对当前状态的增量再喂给模型；
    # 夹爪维度始终保留为绝对值。
    use_delta_joint_actions: bool = True
    # 若观测里没有 prompt key，注入这里的默认任务指令
    default_prompt: str | None = None
    # 若为 True，把标准 ALOHA 数据的关节/夹爪数值换算成 pi 基座模型训练时
    # 使用的内部空间。使用标准 ALOHA 数据训练/微调时建议打开。
    adapt_to_pi: bool = False

    # 默认 repack：数据集的 top 相机、state、action 映射成
    # images.cam_high / state / action。若任务相机更多，可像 TRON2
    # 那样在创建 TrainConfig 时传入自定义 repack_transforms 覆盖。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 从数据集哪个 key 读取动作序列
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # ALOHA 输入/输出变换（例如手臂坐标符号、夹爪开合口径的换算）
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            # make_bool_mask 生成“哪些维度转 delta”的掩码：ALOHA 双臂
            # 各有 6 个关节维度，夹爪维度保持绝对值。
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 通用“模型变换”由 ModelTransformFactory 按模型类型生成
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTronDataConfig(DataConfigFactory):
    """TRON2 双臂机器人（LeRobot 格式）的数据工厂——本仓库真机任务的主入口。

    与 ALOHA 工厂结构类似，区别在于：
      - 使用 Tron2Inputs/Tron2Outputs 做 TRON2 特有的坐标/夹爪变换；
      - state_dim 同时决定模型输出的动作/状态维度：
        16 = 双臂 + 夹爪；18 = 额外包含头部关节；
      - delta 掩码按“左臂 7 维 + 右臂 7 维”生成。

    默认 repack 只映射单路 top 相机；TRON2 真机任务通常在构造配置时用
    _tron2_repack_transforms() 覆盖，映射 cam_high/cam_left_wrist/
    cam_right_wrist 三路相机。
    """

    # 是否把关节角转成增量动作（TRON2 公开权重通常关闭，保持绝对角度）
    use_delta_joint_actions: bool = True
    # 观测里缺 prompt 时注入的默认任务指令
    default_prompt: str | None = None
    # 是否换算到 pi 基座模型内部空间（TRON2 公开权重通常关闭）
    adapt_to_pi: bool = False
    # 状态/输出维度：16 = 双臂 + 夹爪；18 = 再加头部关节
    state_dim: int = 16

    # 默认 repack（单相机/通用字段映射）
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 从数据集哪个 key 读取动作序列
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # TRON2 输入/输出变换；output_dim 用于把模型输出裁剪到 state_dim
        data_transforms = _transforms.Group(
            inputs=[tron2_policy.Tron2Inputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[tron2_policy.Tron2Outputs(adapt_to_pi=self.adapt_to_pi, output_dim=self.state_dim)],
        )
        if self.use_delta_joint_actions:
            # TRON2 双臂各有 7 个关节维度；delta 掩码覆盖两组关节
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """LIBERO 仿真任务（LeRobot 格式）的数据工厂。

    这个类同时充当“自定义数据集教程”：想接入自己的数据时，复制本类，再按
    create() 里的步骤（repack -> data 变换 -> 可选 delta -> 模型变换）
    改成你自己的 key 与变换即可。
    """

    # 旧版 pi0 checkpoint 曾在“数据集本身已是 delta 动作”的基础上再套一层
    # delta 变换（为兼容当时的输入约定）。LIBERO 新配置用 False。
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 第 1 步：repack 变换。它只作用于“来自数据集的样本”，推理时不会执行。
        # 目的：让数据集字段与推理环境尽量一致。下面把数据集的 key（定义在
        # 数据转换脚本里）对应到推理管线使用的 key。
        # 自定义数据集时：先弄清推理环境会传哪些 key，再修改这里的映射；
        # RepackTransform 只重命名 key，不改变数值。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 第 2 步：data transforms，训练与推理都会生效。
        # inputs 是进入模型前的变换；outputs 是模型输出后的还原（仅推理用）。
        # LIBERO 的变换定义在 libero_policy.py；自定义数据集时把它们替换成
        # 你自己的输入/输出变换即可。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 第 3 步（可选）：pi0 模型按 delta 动作（相对动作块首帧）训练。
        # 若数据集动作是绝对值（如目标关节角），需要转成 delta 再训练；
        # 夹爪动作永远保留绝对值。下方掩码把前 6 维（关节）转 delta、
        # 第 7 维（夹爪）保持绝对值。
        # LIBERO 原始数据本身已经是 delta，平时不需要额外转换；只有部分老
        # pi0 checkpoint 需要再套一次 delta（通过 extra_delta_transform 打开）。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 第 4 步：模型变换（prompt 分词、图像 resize、state/action 补齐等）。
        # 自定义数据集通常不需要改这里。
        model_transforms = ModelTransformFactory()(model_config)

        # 组装基础配置 + 三类变换后返回最终 DataConfig
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """DROID 大规模数据（RLDS 格式）的数据工厂。

    当 DROID 数据达到数十至上百小时时，改用 RLDS（TensorFlow Dataset 格式）
    流式加载，训练效率远高于逐个文件读取的 LeRobot 格式。
    注意：使用 RLDS 加载器时必须设置 num_workers=0。
    """

    # RLDS 数据集根目录（必填；create() 末尾有 assert 强制检查）
    rlds_data_dir: str | None = None
    # DROID 动作空间：例如 JOINT_POSITION 表示动作是关节绝对位置
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # 数据过滤选项：可传一个 JSON 文件路径，把 episode 映射到要保留的
    # 时间步区间 (start, end)。episode 的 id 形如
    # "{recording_folderpath}--{file_path}"。

    # 采样的数据集列表：name（名字）、version（版本）、weight（采样权重）、
    # 可选 filter_dict_path（过滤文件）
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # key 重命名：把 DROID RLDS 原始字段对齐到 openpi 通用格式
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # DROID 输入/输出变换（坐标符号、夹爪等约定）
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # RLDS 返回的是关节绝对位置动作 -> 训练前先转成 delta 动作
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        # RLDS 加载器必须指定数据目录，否则立即报错提醒用户
        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """自定义 DROID 数据集（LeRobot 格式）的数据工厂。

    适合规模较小（数十小时以内）的 DROID 数据：先用
    examples/droid/convert_droid_data_to_lerobot.py 转成 LeRobot 格式，
    再在这里设置你的 repo_id 进行微调。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # key 重命名：保留外置双相机 + 手腕相机的字段
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 转换脚本输出的动作是关节*速度*，本身就带增量语义，
        # 因此这里不再额外套 delta 变换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# ============================================================================
# TrainConfig：一次训练/推理任务的“总配置”
# ============================================================================
# TrainConfig 把模型、权重初始化、数据工厂、优化器、训练步数等全部打包成
# 一个对象。_CONFIGS 里每个命名配置（debug、pi0_aloha_sim、pi05_tron2_*）
# 都是它的一个实例；scripts/train.py 等入口消费的就是这个对象。
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置名，必须全局唯一，用于 CLI 子命令与 checkpoint 目录命名
    name: tyro.conf.Suppress[str]
    # 项目名（主要用于 wandb 分组）
    project_name: str = "openpi"
    # 实验名：会拼进 checkpoint/日志目录（{checkpoint_base_dir}/{name}/{exp_name}）。
    # tyro.MISSING 表示命令行必须提供；未设置时 checkpoint_dir 属性会报错。
    exp_name: str = tyro.MISSING

    # 模型配置：决定网络结构、动作维度、动作长度等。
    # 公共属性（action_dim、action_horizon、max_token_len）见 BaseModelConfig；
    # 具体模型（如 Pi0Config）会再增加自己的字段。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 权重加载器：模型初始化后可选地（部分）加载磁盘/云端权重，
    # 微调时用它加载 pi0 基座权重；默认为 NoOpWeightLoader（不加载）。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选：PyTorch 训练入口（scripts/train_pytorch.py）使用的权重路径
    pytorch_weight_path: str | None = None

    # PyTorch 训练的精度（JAX 训练使用自己的 bfloat16 转换逻辑）
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # 学习率计划与优化器配置（见 optimizer.py：CosineDecaySchedule / AdamW 等）
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # EMA（指数移动平均）衰减系数；None 表示关闭 EMA
    ema_decay: float | None = 0.99

    # 冻结过滤器：指定哪些参数不参与训练。
    # 例如 LoRA 微调时把完整模型参数冻结、只训练 adapter。
    # Suppress 表示该字段不直接由命令行指定（通过模型配置生成）。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 训练数据（注意类型是“工厂”）：训练时调用 data.create(...) 得到 DataConfig
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # assets（归一化统计量等）的根目录与 checkpoint 的根目录
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    # 随机种子（数据打乱、权重初始化等）
    seed: int = 42
    # 全局 batch size（所有设备加在一起的大小）
    batch_size: int = 32
    # 数据加载 worker 进程数。调大能加速取数据，但会占用更多 CPU/内存。
    # 注意：RLDS（DROID 全量数据）要求为 0。
    num_workers: int = 2
    # 总共训练多少步（每步消费一个 batch）
    num_train_steps: int = 30_000

    # 每隔多少步打印训练指标 / 保存一次 checkpoint
    log_interval: int = 100
    save_interval: int = 1000
    # 保存 checkpoint 时，step % keep_period == 0 的旧文件不会被清理。
    # 例如 keep_period=5000 表示每 5000 步的 checkpoint 长期保留。
    keep_period: int | None = 5000

    # overwrite=True：目标 checkpoint 目录已存在时直接覆盖；
    # resume=True：从最新 checkpoint 继续训练。二者不能同时为 True
    # （见下方 __post_init__ 的校验）。
    overwrite: bool = False
    resume: bool = False

    # 是否把训练指标同步到 wandb（wandb 需要自行登录）
    wandb_enabled: bool = True

    # 随 policy 一起发布给客户端的元数据（例如 reset_pose、state_dim）。
    # 部署/推理时，scripts/serve_policy.py 会把这里的 metadata 发给客户端。
    policy_metadata: dict[str, Any] | None = None

    # FSDP（全分片数据并行）分片数。>1 时把模型切到指定数量的设备上以降低
    # 单卡显存；>1 会略微降低训练速度。示例：共 4 张卡、fsdp_devices=2，
    # 则模型切成 2 份放在 2 组设备上，组间做数据并行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """本配置自己的 assets 目录：{assets_base_dir}/{config.name}。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """本配置的 checkpoint 输出目录：{checkpoint_base_dir}/{name}/{exp_name}。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """需要训练的参数 = 全部参数 - 被冻结的参数。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        # dataclass 自动调用：在对象创建完成后做一致性校验
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ============================================================================
# TRON2 注册表辅助函数
# ============================================================================
def _tron2_repack_transforms() -> _transforms.Group:
    """TRON2 真机任务专用的 repack：把 LeRobot 三路相机映射到策略输入。

    默认 LeRobotTronDataConfig 只映射单路 top 相机；TRON2 真机任务需要
    cam_high / cam_left_wrist / cam_right_wrist 三路相机，因此在此覆盖。
    """
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )


def _tron2_task_config(name: str, repo_id: str, default_prompt: str) -> TrainConfig:
    """创建单个 pi0.5-TRON2 真机任务配置。

    公开 TRON2 任务共享同一套设定：pi0.5 模型、pi05 基座权重初始化、
    绝对关节角（不转 delta）、20000 训练步；差别只有任务名 / repo_id /
    默认 prompt。
    """
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotTronDataConfig(
            repo_id=repo_id,
            default_prompt=default_prompt,
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repack_transforms=_tron2_repack_transforms(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    )


def _tron2_checkpoint_task_configs() -> list[TrainConfig]:
    """一次性生成所有公开 TRON2 任务配置（alarm ... sort）。

    调用处用列表前的 * 把这些配置“展开”合并进 _CONFIGS。
    """
    return [
        _tron2_task_config("pi05_tron2_alarm", "alarm", "Perform the TRON2 alarm task"),
        _tron2_task_config("pi05_tron2_banana", "banana", "Perform the TRON2 banana task"),
        _tron2_task_config("pi05_tron2_cabinet", "cabinet", "Perform the TRON2 cabinet task"),
        _tron2_task_config("pi05_tron2_candy", "candy", "Perform the TRON2 candy task"),
        _tron2_task_config("pi05_tron2_chess", "chess", "Perform the TRON2 chess task"),
        _tron2_task_config("pi05_tron2_cloth", "cloth", "Perform the TRON2 cloth task"),
        _tron2_task_config("pi05_tron2_drawer", "drawer", "Perform the TRON2 drawer task"),
        _tron2_task_config("pi05_tron2_duck", "duck", "Perform the TRON2 duck task"),
        _tron2_task_config("pi05_tron2_sort", "sort", "Perform the TRON2 sort fruit task"),
    ]


# ============================================================================
# 注册表 _CONFIGS：所有“开箱即用”的命名配置
# ============================================================================
# _CONFIGS 是这个文件的“货架”：cli()（命令行子命令）与 get_config()
# （按名字取配置）都基于它。新增配置的方法：
#   * 直接向列表追加一个 TrainConfig(...)；
#   * 或像 TRON2 / RoboArena / PolaRiS 那样，先写一个返回 list 的函数，
#     再用列表前的 * 把它展开合并进来。
# 注意最后有全局唯一性校验：配置名不能重复。
_CONFIGS = [
    #
    # —— 推理（Inference）用 ALOHA 配置：pi0 / pi0.5 官方基座权重 ——
    # 这些配置主要配合 scripts/serve_policy.py 做部署推理，未指定真实
    # 数据集 repo_id，因此不会用于训练。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # —— 推理用 DROID 配置：pi0 / pi0-FAST / pi0.5 三种架构 ——
    # 使用 SimpleDataConfig + DROID 变换，prompt 从数据集 task 字段读取。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # —— LIBERO 微调配置（“如何为自定义数据集微调”的最佳教程）——
    # 下面几个 TrainConfig 演示了微调基座模型所需的关键要素：用哪个数据集、
    # 初始化自哪个基座 checkpoint、训练多少步 / 用什么学习率等。
    # 新手可以照着 pi0_libero 的结构改 repo_id / 数据工厂 / 超参数，
    # 得到自己的微调配置。
    TrainConfig(
        # 修改 name，让它体现你的模型和数据集。
        name="pi0_libero",
        # 模型配置：这里用 pi0 架构做“全量微调”。下面两个例子分别演示
        # 如何改成 LoRA 低显存微调、以及如何换成 pi0-FAST 架构。
        model=pi0_config.Pi0Config(),
        # 数据集：这里用 LIBERO。自定义时把 repo_id 换成自己的数据集，
        # 并把 data=... 换成上面为你的数据集编写的工厂类。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # prompt_from_task=True：从 LeRobot 数据集自带的 task 字段读取
                # 任务指令，写入输入字典的 prompt 字段。推荐打开。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 初始化权重：加载哪个预训练 checkpoint（必须与上面的模型架构匹配），
        # 这里是 pi0 基座模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 其余超参数（学习率、训练步数等）都可以在这里覆盖；
        # 可用的完整字段列表见上面 TrainConfig 类。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # LoRA 微调示例：把主干换成带 LoRA 的小尺寸变体，显存占用更低
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # freeze_filter 决定训练时冻结哪些参数（全模型冻结，只训 LoRA）。
        # 模型配置提供了配套的 get_freeze_filter() 帮助函数，只需保证它和
        # 上面 model=... 的配置完全一致。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # LoRA 微调时通常关闭 EMA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # pi0-FAST 全量微调示例。
        # action_dim / action_horizon 要和你的数据集一致（action_horizon =
        # 期望一次输出的动作块长度）。
        # max_token_len 是模型能处理的最大（非图像）token 数，包含分好词的
        # prompt、本体状态和（FAST 离散化后的）动作 token：
        #   - 设太小会截断序列尾部（训练时会打警告）；
        #   - 设太大会浪费显存（batch 内每条样本都会补齐到该长度）。
        # 经验值：单臂机器人约 180，双臂约 250。建议先取较小值，若训练中
        # 频繁出现截断警告再逐步调大。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 注意：pi0-FAST 要加载 pi0-FAST 的基座 checkpoint，不能混用 pi0。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # pi0-FAST + LoRA 的低显存微调示例；
        # action_dim / action_horizon / max_token_len 的取值说明见上方
        # pi0_fast_libero 的注释。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 提取冻结过滤器时，同样要保证与上面的模型配置完全一致。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # LoRA 微调时通常关闭 EMA
        ema_decay=None,
    ),
    TrainConfig(
        # pi0.5-LIBERO：更大的 batch、更长的学习率预热/衰减，并使用 EMA。
        # pytorch_weight_path 只是占位示例路径；用 PyTorch 入口训练前
        # 需改成你实际的权重路径。
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # —— ALOHA 微调配置（自定义 LeRobot 数据集示例）——
    # 以 pi0/pi0.5 在 pen_uncap 数据上的微调为例，展示如何加载基座 assets、
    # 如何用默认 prompt、如何自定义三相机 repack。自定义 ALOHA 数据集的
    # 转换与训练步骤见 examples/aloha_real/README.md。
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "action": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # —— TRON2 任务配置（pi0.5 微调）——
    # pi05_tron2_example 是新任务模板；日常新增 TRON2 任务推荐用 YAML 入口，
    # 无需改本文件：
    #   cp configs/train/tron2_tasks/example.yaml \
    #      configs/train/tron2_tasks/<task>.yaml
    #   uv run scripts/train_tron2_task.py \
    #      --task-config configs/train/tron2_tasks/<task>.yaml
    # 下面 *_tron2_checkpoint_task_configs() 用 * 展开注册公开任务
    # （pi05_tron2_alarm ... pi05_tron2_sort）。
    #
    TrainConfig(
        name="pi05_tron2_example",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotTronDataConfig(
            repo_id="example_task",
            default_prompt="Perform the configured TRON2 manipulation task",
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repack_transforms=_tron2_repack_transforms(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    *_tron2_checkpoint_task_configs(),
    #
    # —— DROID 微调配置 ——
    # 包含两类：全量 DROID（RLDS 高效加载，数十至上百小时数据）与
    # 自定义小规模 DROID（LeRobot 格式）。
    TrainConfig(
        # 在 DROID 全量数据上微调 pi0-FAST 基座。
        # 数据量很大，因此用 RLDS 格式做高效流式加载。
        # 自定义（小规模）DROID 数据的写法见下面的 pi05_droid_finetune。
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 改成你本机 DROID RLDS 数据集的路径（`droid` 目录的上一级目录）。
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 10 万步在 8x H100 上约需 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 重要：RLDS 加载器要求 num_workers=0（它内部自行并行）
    ),
    TrainConfig(
        # 在 DROID 全量数据上微调 pi0.5（同样使用 RLDS 加载）。
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 改成你本机 DROID RLDS 数据集的路径（`droid` 目录的上一级目录）。
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # 重要：RLDS 加载器要求 num_workers=0（它内部自行并行）
    ),
    TrainConfig(
        # 在自定义（小规模）DROID 数据上微调 pi0.5 的示例。
        # 这里和大多数微调示例一样使用 LeRobot 格式；若你的自定义数据
        # 在 10 小时以内，可先用 examples/droid/convert_droid_data_to_lerobot.py
        # 把它转成 LeRobot 格式。
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi0.5 使用 32 维动作
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # 换成你自己的 DROID LeRobot 数据集 repo id。
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 重要：微调时必须复用原版 DROID 数据的归一化统计量！
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # —— ALOHA 仿真训练配置 ——
    # pi0_aloha_sim 演示在简单仿真环境（aloha_sim_transfer_cube_human）上
    # 训练，是新手最容易跑通的“真实数据训练”起点。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # —— 调试/冒烟测试配置 ——
    # 使用假数据 + 极小的 dummy 模型，几分钟内验证训练代码能跑通；
    # debug_restore 额外验证“从已有 checkpoint 继续训练”的流程。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # —— 其他仓库模块注册的配置（RoboArena / PolaRiS）——
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

# 唯一性校验：配置名重复会让 cli()/get_config() 产生歧义，直接启动报错。
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

# 把列表转成 dict：{配置名 -> TrainConfig}，O(1) 按名查找。
# 这一行放在唯一性校验之后，保证 key 不会相互覆盖。
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    """命令行入口：先选择一个配置名，再按字段覆盖任意超参数。

    tyro 会为每个配置名生成一个“子命令”，例如：
        python scripts/train.py debug --exp-name smoke_test

    子命令内部暴露该配置的所有字段（batch-size、num-train-steps 等），
    没覆盖的字段使用注册表里的默认值。
    """
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名字获取配置（代码内调用，与 CLI 无关）。

    Args:
        config_name: 注册表里的配置名，例如 "pi05_tron2_candy"。

    Returns:
        对应的 TrainConfig 对象。

    Raises:
        ValueError: 名字不存在；若存在相近名字会给出“你是不是想找 X”提示。
    """
    if config_name not in _CONFIGS_DICT:
        # difflib.get_close_matches 找出拼写最接近的配置名（cutoff=0 表示
        # 任何相似度都会返回，最多 1 个），让新手能立刻发现拼写错误
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
