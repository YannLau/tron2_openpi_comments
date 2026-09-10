"""JAX / Flax NNX 训练入口：读取一个 TrainConfig，
然后在多设备（单机多卡或多机）上训练 pi0 系列模型。

快速上手（Quick Start）
-----------------------
    # 1) 查看有哪些已注册的配置（配置名单见 src/openpi/training/config.py 的 _CONFIGS）
    uv run scripts/train.py --help

    # 2) 用某个配置开始训练（exp_name 必填，决定 checkpoint / 日志的输出目录）
    uv run scripts/train.py pi05_tron2_candy --exp-name my_first_run

    # 3) 断点续训 / 覆盖已有实验目录
    uv run scripts/train.py <config> --exp-name my_first_run --resume
    uv run scripts/train.py <config> --exp-name my_first_run --overwrite

命令行解析由 training/config.py 的 cli() 负责：配置名是“子命令”，其余字段可用
--batch-size / --num-train-steps 等覆盖。PyTorch 训练入口是 scripts/train_pytorch.py。

main() 的执行流程（建议按这个顺序阅读）
---------------------------------------
 1. init_logging：设置日志格式；
 2. 校验 batch_size 能被设备数整除，并配置 JAX 编译缓存目录；
 3. 由 seed 拆出 train_rng / init_rng，由 fsdp_devices 建立设备网格 mesh；
 4. 初始化 checkpoint 目录（新建 / 覆盖 / 续训）与 wandb；
 5. 创建数据加载器，先取一个 batch 用于日志与图像 sanity check；
 6. init_train_state：随机初始化模型 → 可选地加载预训练权重 → 计算参数分片方式
    （若 --resume，则这里只算“形状骨架”，真实参数稍后从 checkpoint 恢复）；
 7. 用 jax.jit 包装 train_step，进入训练循环：前向 → 反向 → 优化器更新 → 记录指标；
 8. 按 save_interval 保存 checkpoint，最后等待异步保存完成再退出。

理解本文件需要的几个概念
------------------------
- TrainState（见 training/utils.py）：训练状态 = 参数 params + 模型结构 model_def +
  优化器状态 opt_state + 优化器定义 tx + EMA 参数；
- nnx.split / nnx.merge：把模型拆成“结构（graphdef） + 状态（state）”，
  这样参数就能当普通 PyTree 处理（加载权重、分片、在 jit 之间传递）；
- mesh / sharding：把设备组织成二维网格（batch 方向的数据并行, fsdp 方向的模型切分）。
  数据按 data_sharding 分片；优化器状态等小数组用 replicated_sharding 在每个设备各存一份；
- jax.eval_shape：只推导形状、不做真正计算。用它拿到 train_state 的“形状骨架”后，
  既能据此决定权重加载的目标结构，也能据此计算分片方案；
- 冻结参数（freeze_filter）：LoRA 微调时只训练少量适配器，其余参数冻结并转成 bfloat16。

新手小贴士
----------
- 日志里的 loss / grad_norm / param_norm 分别反映“学得好不好 / 梯度是否爆炸 / 权重尺度”；
- 第一次运行会触发大量 JIT 编译，比较慢；编译结果会缓存到 ~/.cache/jax 供下次复用；
- 想快速跑通流程，可参考 config.py 里的 debug 配置，并把 num_train_steps、batch_size 调小。
"""

# --- 第三方 / 框架库 ---
import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath  # 路径工具，支持 gs:// 等；这里用来定位代码仓库根目录
import flax.nnx as nnx  # Flax 新神经网络 API：模型结构、参数状态与自动微分
from flax.training import common_utils  # stack_forest：把多步指标堆叠起来求均值
import flax.traverse_util as traverse_util  # PyTree 拍平 / 还原（过滤形状占位符时用到）
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax  # 优化器与梯度变换（AdamW 等）
import tqdm_loggable.auto as tqdm  # 在 Jupyter / 终端都能正常刷新的进度条
import wandb  # 实验跟踪（指标曲线、代码快照）

# --- 本仓库内部模块 ---
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    # 自定义日志格式：默认的级别名（WARNING 等）太长，这里缩成一个字母，
    # 让每行日志更紧凑、方便在终端里扫读。
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # 只改 record.levelname 的“显示值”，不影响日志级别本身的判断逻辑。
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        # 各字段含义：时间.毫秒 [级别] 消息（左对齐、最少 80 字符） (进程号:文件名:行号)
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 输出 INFO 及以上级别
    # 注意：这里假设根 logger 已经至少有一个 handler（导入的第三方库会创建），
    # 本函数只负责把它的输出格式换掉。
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 Weights & Biases 实验跟踪。

    Args:
        config: 训练配置，提供 project_name / exp_name / checkpoint_dir 等信息。
        resuming: True 表示这是断点续训，要接续到同一条 wandb 曲线。
        log_code: 是否把代码上传到 wandb（默认关闭，可选）。
        enabled: False 时彻底禁用 wandb（例如离线跑 smoke test）。
    """
    if not enabled:
        # 用 disabled 模式初始化：之后代码里所有 wandb.log 都会自动变成空操作，
        # 不必在训练循环里到处写 if wandb_enabled。
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        # checkpoint 目录应当由 initialize_checkpoint_dir（main 里更早调用）创建好，
        # 这里只是防御性检查，避免把 wandb id 写到不存在的目录。
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        # 续训：读回上次保存的 run id，并用 resume="must" 强制接到同一个 run；
        # 如果 id 不存在，wandb 会直接报错而不是悄悄新建一条曲线。
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新实验：把整个 TrainConfig 作为超参数记录下来，
        # 并把本次 run id 写进 checkpoint 目录，供以后 --resume 使用。
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        # 上传仓库源码快照（__file__ 在 scripts/ 下，parent.parent 是仓库根目录）。
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    # 加载 / 合并外部权重（具体策略见 training/weight_loaders.py：
    # NoOp = 不加载；Checkpoint = 加载某个 checkpoint；PaliGemma = 只加载 VLM 主干）。
    # 传入的 params_shape 来自 jax.eval_shape，叶子是“只有形状 / dtype”的 ShapeDtypeStruct。
    loaded_params = loader.load(params_shape)
    # 校验加载器返回值与目标结构完全一致（结构、形状、dtype 都要对得上）：
    # 加载器如果漏掉了某些 key，会在这里得到一条清晰的报错信息。
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    # 过滤掉 ShapeDtypeStruct 叶子：它们只是“形状占位符”，并不携带真实数值；
    # 剩下的叶子才是真正从 checkpoint 读出来的权重。
    # 结果可能只是全部参数的一个子集（例如微调时只加载基座权重，LoRA 用随机初始化）。
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """创建（或准备恢复）训练状态，并计算它的分片方案。

    返回值是 `(train_state, sharding)`：
    - 非续训时，train_state 是参数已就绪、可以直接训练的实例；
    - 续训时，train_state 只是“形状骨架”（叶子为 ShapeDtypeStruct），
      真实的 step / params / opt_state 由 checkpoints.restore_state 从磁盘恢复。
    """
    # 优化器：由配置里的学习率计划 + 优化器类型构造出 optax 的“梯度变换”。
    # 注意 weight_decay_mask=None 表示对全部参数统一处理，不做分组权重衰减。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """完整的初始化逻辑：随机建模型 → 写入加载的权重 → 组装 TrainState。

        这个函数会先被 jax.eval_shape 调用一次（只推形状），
        再被 jax.jit 调用一次（真正执行）；因此这里必须保持纯函数风格。
        """
        # 拆出用于模型初始化的随机数（这里第一个返回值未再使用）。
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            # NNX 模型 = 结构(graphdef) + 状态(state)：
            # 先拆开，用加载到的权重替换 state 中对应位置，再合并回完整模型。
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            # 若 partial_params 里出现了模型没有的 key，这里会直接报错（防止加载错配置的权重）。
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)  # 取出模型的全部参数 / 状态（nnx.State 是一种 PyTree）
        # Convert frozen params to bfloat16.
        # 冻结参数（如预训练主干）不参与训练，转成 bfloat16 可以省下一半显存。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,  # 已训练步数，从 0 开始
            params=params,  # 模型参数（nnx.State）
            model_def=nnx.graphdef(model),  # 模型结构（不含参数），用于后续 nnx.merge 还原模型
            tx=tx,  # optax 梯度变换（优化器定义）
            # 优化器状态只为“可训练参数”初始化，冻结参数不产生状态，省显存。
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,  # EMA 衰减系数；None 表示关闭 EMA
            ema_params=None if config.ema_decay is None else params,  # EMA 参数从初始参数开始
        )

    # jax.eval_shape 不真正执行 init，只返回“形状骨架”：
    # 叶子从真实数组变成 jax.ShapeDtypeStruct（含 shape / dtype），几乎不占内存。
    train_state_shape = jax.eval_shape(init, init_rng)
    # 根据 mesh 为骨架里的每个数组决定分片方式：大矩阵按 fsdp 轴切分，小数组 / 向量复制。
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 续训：把“形状骨架 + 分片方案”返回给 main，真实参数稍后由 restore_state 从磁盘读入。
        return train_state_shape, state_sharding

    # 非续训：按 weight_loader 的策略加载权重（可能只是部分参数），
    # 目标是前面 eval_shape 得到的参数结构（只有形状，没有数值）。
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    # 加载权重时先放在“每台设备各一份”的分片上，避免下载 / 读取时就做复杂切分。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    # 用 jit 执行 init：
    # - in_shardings：init_rng 与 partial_params 都是复制到所有设备的；
    # - out_shardings：输出的 TrainState 按前面算好的 state_sharding 分布到各设备；
    # - donate_argnums=(1,)：partial_params 传入后不再需要，允许 JAX 回收它的显存。
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """执行一步训练：前向计算损失 → 反向求梯度 → 优化器更新参数。

    本函数会被 jax.jit 整体编译（见 main 里的 ptrain_step），
    因此其中不能有依赖 Python 状态的副作用；`config` 作为静态参数被烘焙进编译结果。
    """
    # 从“结构 + 参数”还原出可调用的模型对象；model.train() 切换到训练模式
    # （例如启用 dropout，pi0 里对应 deterministic=False）。
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # compute_loss 返回“逐样本 / 逐动作步”的损失，这里取均值变成标量损失。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 用 fold_in(rng, step) 让每一步的随机噪声不同，同时保证“同种子可复现”。
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    # DiffState(0, filter)：告诉自动微分“只对第 0 个参数（model）中匹配 filter 的参数求梯度”。
    # 冻结参数既不计算梯度、也不更新，从而大幅节省显存和计算量（LoRA 微调的关键）。
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # value_and_grad：一次前向同时返回 loss 和梯度（反向传播）。
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)  # 只取出可训练参数
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)  # 优化器算出参数更新量
    new_params = optax.apply_updates(params, updates)  # 施加更新量，得到新的可训练参数

    # Update the model in place and return the new full state.
    # 把新参数写回模型（只更新可训练部分，冻结参数保持原值），
    # 再取回完整参数状态，以维持 TrainState.params 的完整结构。
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # dataclasses.replace：基于旧 state 生成新 state（TrainState 是不可变的 dataclass）。
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # EMA（指数移动平均）：ema = decay * 旧EMA + (1 - decay) * 当前参数。
        # 推理 / 评估时常用 EMA 参数，曲线更平滑、通常泛化更好。
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    # 下面的 param_norm 只统计“权重核（kernel）”：排除 bias / scale / 位置与输入嵌入，
    # 并且只保留 ndim > 1 的矩阵。这样得到的范数更能反映网络主体的尺度变化。
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # 训练指标：loss（损失）、grad_norm（梯度全局范数，爆炸报警）、
    # param_norm（权重核范数，观察权重是否异常增长）。
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """训练主流程：准备环境 → 初始化状态 → 循环训练 → 保存 checkpoint。

    阅读时可把本函数当成“目录”：每一段注释对应上面模块 docstring 里的一个步骤，
    细节实现分散在 init_train_state / train_step / checkpoints / data_loader 等模块。
    """
    init_logging()
    logging.info(f"Running on: {platform.node()}")  # 打印机器名，方便多机训练时分辨日志来源

    # batch_size 要平均分给所有设备（数据并行 / FSDP 都要求能整除），否则无法切分。
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 把 JIT 编译产物缓存到磁盘：第一次训练很慢，之后重跑可以跳过重复编译。
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 从配置的 seed 派生随机数：
    #   train_rng —— 训练循环用（每步再 fold_in(step)，保证可复现）；
    #   init_rng  —— 模型参数的随机初始化用。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 建立二维设备网格：(batch 方向的数据并行, fsdp 方向的模型切分)。
    # config.fsdp_devices=1 时等价于纯数据并行。
    mesh = sharding.make_mesh(config.fsdp_devices)
    # 数据 sharding：batch 维同时沿 (batch, fsdp) 两个轴切分，让每台设备拿到不同的样本。
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 复制 sharding：空的 PartitionSpec 表示每个设备各存一份（用于标量 / 小数组 / 优化器状态）。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化 checkpoint 管理器：
    # - 目录已存在且 overwrite=True → 清空重建；
    # - 目录已存在且 resume=True → 返回 resuming=True，稍后恢复训练状态；
    # - 都没有指定却已存在 → 报错，避免误覆盖别人的实验。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    # wandb 需要在 checkpoint 目录存在之后初始化（resume 时它要从目录里读 run id）。
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建训练数据加载器。shuffle=True 表示每个 epoch 打乱样本顺序；
    # sharding=data_sharding 表示数据加载时就按 mesh 分好片，训练时无需再搬运。
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    # 先取一个 batch：一方面做日志 / 可视化 sanity check，另一方面让数据 pipeline 预热。
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if config.wandb_enabled:
        # 记录前 5 个样本的所有相机视图（横向拼接成一张图），
        # 用来肉眼确认图像数据、裁剪 / resize 是否正确。
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态（随机权重 + 可选加载预训练权重 + 计算分片方案）。
    # resume 时这里只返回“形状骨架”，真实参数在下一行从 checkpoint 恢复。
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # 等待初始化真正执行完成（JAX 是异步派发的），这样后面的日志才能拿到真实数值。
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        # 从最新 checkpoint 恢复 step / params / opt_state / EMA 等，接着上次继续训练。
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 把 train_step 编译成一个可调用的函数：
    # - functools.partial 把 config 固定为静态参数（编译期常量，不参与 jit 追踪）；
    # - in_shardings / out_shardings 必须与实际的 3 个输入、2 个输出一一对应；
    # - donate_argnums=(1,) 表示旧 train_state 的缓冲区可以被回收，
    #   因为下一步我们会用新 state 覆盖它（省显存，是大模型训练的常用技巧）。
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 从 TrainState 里取出当前步数（续训时可能 > 0），进度条与循环都从它开始。
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # set_mesh：让模型内部 activation_sharding_constraint 能拿到当前 mesh，
        # 从而把中间激活也按 DATA_AXIS 切分（见 training/sharding.py 的说明）。
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        # 累积每一步的指标；到 log_interval 时求平均再打印，避免日志刷屏。
        infos.append(info)
        if step % config.log_interval == 0:
            # stack_forest：把 list[dict] 变成 dict[stacked arrays]，再逐项求均值；
            # jax.device_get 把结果从设备拷回 CPU，才能格式化打印 / 传给 wandb。
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        # 取出下一个 batch（训练用的是循环开头传入的“当前 batch”）。
        # 数据加载器内部有多个 worker 进程在后台预取，这里提前取下一批
        # 可以尽量避免训练循环因数据未就绪而停顿。
        batch = next(data_iter)

        # 周期性保存 checkpoint（跳过起始步，避免续训时立刻重复保存），
        # 并在最后一步强制保存一次，确保训练结果落盘。
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    # 保存是异步的：退出前必须等待所有写入完成，否则进程结束可能丢掉最后的 checkpoint。
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # _config.cli()：用 tyro 解析命令行。第一个参数是“配置名”（子命令），
    # 其余 --xxx 覆盖该配置的字段；解析结果是一个 TrainConfig 对象。
    main(_config.cli())
