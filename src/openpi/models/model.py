"""模型核心定义：观测/动作的数据格式、模型基类与配置、参数恢复工具。

本文件是 openpi 训练 / 推理流水线的“类型中枢”。它本身不实现任何具体网络结构，
而是定义所有 pi0 系列模型（pi0 / pi0_fast / pi05）共享的东西：

1. **输入 / 输出数据的标准格式**：`Observation`（观测）、`Actions`（动作）；
2. **输入数据的预处理逻辑**：图片尺寸对齐、训练时数据增强、mask 补全（`preprocess_observation`）；
3. **模型配置与模型基类**：`BaseModelConfig` / `BaseModel`，约定“如何创建 / 加载模型、
   如何计算损失、如何采样动作”这套统一接口；
4. **从 checkpoint 恢复参数的工具函数**：`restore_params`。

整体数据流大致是：

    transforms.py（数据变换）--产出嵌套 dict--> Observation.from_dict() --> Observation
        --> model.compute_loss(rng, obs, actions)   # 训练阶段：计算损失
        --> model.sample_actions(rng, obs)          # 推理阶段：采样动作，输出 Actions

新手阅读建议：
- 先看类型别名 `Actions` 和类 `Observation`，理解输入输出的“形状约定”（batch/h/w/c、s、l、ah、ad 各指什么）；
- 再看 `BaseModelConfig` / `BaseModel` 的接口方法，理解“一个模型要接入这套框架需要实现哪些方法”；
- 最后看 `preprocess_observation` 与 `restore_params`，它们是数据端和权重端两个通用工具。
"""

import abc  # Python 抽象基类库，用 @abc.abstractmethod 声明“子类必须实现”的抽象方法
from collections.abc import Sequence  # 通用序列类型（list/tuple 等），用于类型标注
import dataclasses  # dataclass 工具，自动生成 __init__/__repr__/__eq__ 等样板代码
import enum  # 枚举库，用于定义 ModelType
import logging  # 日志库，本文件统一用 logger = logging.getLogger("openpi") 输出日志
import pathlib  # 路径库，用于处理 checkpoint 路径
from typing import Generic, TypeVar  # 泛型支持，用于给 Observation 打上“元素数组类型”参数 ArrayT

import augmax  # 纯 JAX 实现的图像增强库（随机裁剪、旋转、颜色抖动等），仅在训练时使用
from flax import nnx  # Flax 的新版神经网络 API（NNX）：用 nnx.Module / nnx.split / nnx.merge 管理模型参数
from flax import struct  # Flax 的 dataclass 封装，使 dataclass 能被当作 JAX PyTree 参与 jit/vmap/tree.map
from flax import traverse_util  # PyTree 遍历工具，这里用于扁平化 / 还原嵌套参数 dict
import jax  # 谷歌的函数式数组计算框架（类似可求导的 numpy，运行在 GPU/TPU）
import jax.numpy as jnp  # JAX 版 numpy，数组运算入口
import numpy as np  # CPU 上的 numpy
import orbax.checkpoint as ocp  # JAX 生态的 checkpoint 读写库
import safetensors  # 安全的模型权重存储格式（比 pickle 更安全、读写更快）
import torch  # PyTorch，用于加载 PyTorch 版模型权重（load_pytorch）

from openpi.models_pytorch import pi0_pytorch  # PyTorch 实现的 pi0 模型（推理 / 部署场景使用）
from openpi.shared import image_tools  # 图像工具，如 resize_with_pad（等比缩放 + 黑边填充）
import openpi.shared.array_typing as at  # 本项目统一的数组类型别名与运行时类型检查工具

logger = logging.getLogger("openpi")

# ArrayT 是一个“类型变量”（TypeVar），表示“任意一种数组类型”。
# openpi 同时支持 JAX 数组、PyTorch 张量、numpy 数组三种后端，
# 因此用 ArrayT 把这些类型圈起来，让 Observation 等数据结构可以同时兼容三者，
# 并且能用 jaxtyping 给数组加上“形状注释”。
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types. 支持的模型种类。"""

    PI0 = "pi0"  # 原始 pi0：动作轨迹通过“流匹配（flow matching）/扩散”方式去噪生成
    PI0_FAST = "pi0_fast"  # 加速版 pi0：动作改为“自回归逐 token 预测”，推理更快
    PI05 = "pi05"  # 新一代 pi05 架构


# 模型固定要求输入的 3 路相机图像：
#   - base_0_rgb        ：机身上的主相机（前方 / 全局视角）
#   - left_wrist_0_rgb  ：左机械臂腕部相机
#   - right_wrist_0_rgb ：右机械臂腕部相机
# 数据变换时至少要提供这几个键；多出来的视角会被 preprocess_observation 忽略。
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# 模型统一使用的输入图片分辨率 (宽, 高)。
# 注释说明：如果未来发布小模型，这个值可能还需要调整。
IMAGE_RESOLUTION = (224, 224)


# ========================== 数据格式约定 ==========================
# 数据变换（transforms.py）最终把一条数据变成“嵌套字典”，
# 之后再经 `Observation.from_dict()` 转成结构化的 Observation / Actions 对象。见下文。
#
# 字典形式的数据长这样：
# {
#     # ---- 观测（Observation）部分 ----
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像，数值范围 [-1, 1] 或 [0, 255]
#         ...  # 其它相机视角（数量不限）
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # 该样本图像是否有效（True = 有效）
#         ...  # 各视角对应的 mask
#     },
#     "state": float32[*b, s],  # 低维机器人状态（关节角、末端位姿等）
#     "tokenized_prompt": int32[*b, l],   # 可选：语言指令分词后的 token 序列
#     "tokenized_prompt_mask": bool[*b, l],  # 可选：prompt token 的 mask（如区分真实 token 与 padding）
#     "token_ar_mask": int32[*b, l],     # 可选：pi0_fast 专用，自回归 token mask
#     "token_loss_mask": bool[*b, l],    # 可选：pi0_fast 专用，损失 mask
#
#     # ---- 动作（Actions）部分 ----
#     "actions": float32[*b ah ad]
# }
# 其中符号含义：
#   *b = 任意个批量维度（通常就是一个 batch 维）
#   h, w = 图片高 / 宽
#   s = state 的维度
#   l = 序列长度（token 数量）
#   ah = action_horizon（一次预测多少步未来的动作）
#   ad = action_dim（每步动作的维度）
#
# 补充说明：
#   - 图片默认是 NHWC 布局（batch, height, width, channel）；
#   - 训练时图片用 float32 且值在 [-1, 1]；dataloader 也可能给 uint8 [0,255]，由 from_dict 归一化；
#   - pi0 的动作是“轨迹式”的：一次预测未来连续多步（action_horizon 步），每步 ad 维。
@at.typecheck  # 用 beartype 做运行时类型检查：传入数组的形状 / 类型不匹配会直接报错，方便调试
@struct.dataclass  # Flax 的 dataclass：既像普通 dataclass，又能作为 JAX PyTree 被 jit/vmap/tree.map 处理
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model. 存放观测数据，也就是“喂给模型的输入”。

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    （期望的字典形式见 `Observation.from_dict` 和文件开头的注释，数据变换就应该产出这种格式。）
    """

    # 各相机图像，float32，数值范围 [-1, 1]。
    # 类型注解 at.Float[ArrayT, "*b h w c"] 是 jaxtyping 语法，表达：
    #   这是一个形状为 [*b, h, w, c] 的浮点数组（*b=批量维度，h/w=高/宽，c=通道数）。
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # 与 images 同键的 mask，bool 类型，形状 [*b]，True 表示该样本的图像有效。
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # 低维机器人状态（关节角等），形状 [*b, s]。
    state: at.Float[ArrayT, "*b s"]

    # 语言指令的 token 序列（可选）。pi0 内部用 VLM（PaliGemma）编码语言指令，输入前需先分词。
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # prompt token 的 mask（可选），用于区分真实 token 与 padding。
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # ---- pi0_fast 模型专用字段（其它模型保持 None 即可）----

    # 自回归 token mask：标记每个位置是“已知 / 输入”还是“待生成”的 token（FAST 模型用）。
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # 损失 mask：标记哪些 token 位置参与自回归损失计算（FAST 模型用）。
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format.
        （把“嵌套字典”形式的原始数据转换成结构化的 Observation。

        这是数据变换产出的最终落点：transforms 产出字典，这里负责字段映射 + 图像归一化。）
        """
        # 安全校验：tokenized_prompt 和它的 mask 必须成对出现。
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # 图像归一化：如果图像是 uint8（范围 [0, 255]），统一转成 [-1, 1] 的 float32。
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                # JAX / numpy 数组：线性归一化。/255*2-1 把 [0, 255] 映射到 [-1, 1]。
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                # PyTorch 张量：除了归一化，还要把布局从 NHWC 转成 NCHW——
                # PyTorch 版模型（pi0_pytorch）期望通道前置的 (B, C, H, W) 布局，故 permute(0, 3, 1, 2)。
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        # 用 data.get() 读取可选字段：字典里没提供的自动填 None。
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict. 把 Observation 转回嵌套字典（from_dict 的逆操作）。"""
        result = dataclasses.asdict(self)  # 把 dataclass 展开成普通 dict
        result["image"] = result.pop("images")  # 键名 images -> image，与前面的数据格式约定保持一致
        result["image_mask"] = result.pop("image_masks")  # 键名 image_masks -> image_mask
        return result


# 动作数据格式。数据变换产出的字典里，动作放在 "actions" 键下。
# 形状 [*b, ah, ad]：
#   ah = action_horizon：一次预测的动作步数（未来轨迹长度）；
#   ad = action_dim：每步动作的维度（如 7 个关节角 + 1 个夹爪开合 = 8）。
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,  # JAX 随机数 key。train=False 时可以为 None（不需要随机增强）
    observation: Observation,  # 原始观测：可能分辨率不对、缺少 mask、未做数据增强
    *,
    train: bool = False,  # 是否为训练阶段：True 时做随机数据增强，False 时只做尺寸对齐
    image_keys: Sequence[str] = IMAGE_KEYS,  # 需要处理的相机视角（默认 3 路）
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,  # 目标分辨率 (宽, 高)
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).

    （预处理观测数据：按需做图像增强（训练时）、尺寸对齐（等比缩放 + 填充）、补全 mask。）
    """

    # 1) 检查所需相机视角是否齐全。
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 批量形状（去掉 state 最后一维 s），之后用来构造“全 True”的默认 mask。
    batch_shape = observation.state.shape[:-1]

    # 2) 逐路相机处理图像。
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        # 尺寸对齐：若当前分辨率不是目标分辨率，做“等比缩放 + 黑边填充”，避免图像拉伸变形。
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # ---- 训练阶段的随机数据增强 ----
            # augmax 期望输入值域是 [0, 1]，所以先把 [-1, 1] 的图像平移到 [0, 1]。
            image = image / 2.0 + 0.5

            transforms = []
            # 腕部相机（wrist）不做随机裁剪 / 旋转：手腕视野小，裁切 / 旋转容易丢关键信息；
            # 主相机（base）则做“随机裁剪 + 放大回原尺寸 + 小角度旋转”，增强泛化性。
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),  # 随机裁掉约 5% 边缘
                    augmax.Resize(width, height),  # 裁剪后缩回原尺寸
                    augmax.Rotate((-5, 5)),  # 在 ±5° 内随机旋转
                ]
            # 所有相机统一做颜色抖动（亮度 / 对比度 / 饱和度扰动）。
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            # 每个 batch 样本用独立的随机子 key，再用 vmap 批量并行执行整条增强链（chain）。
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 增强完成后回到模型期望的 [-1, 1] 值域。
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 3) 补全 mask：数据里没提供某路相机的 mask 时，默认全部有效（全 True，即“不做遮挡”）。
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default（默认不做遮挡）
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 4) 组装并返回新的 Observation（state 和 prompt 原样透传，不在本函数内修改）。
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)  # frozen=True：实例创建后不可修改，保证配置不可变、可安全复用
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.

    （所有模型共享的配置基类。具体模型继承本类，并实现 `create` 方法以创建对应的模型。

    例如 pi0 的配置类会继承这里，额外加上 pi0 特有的超参数（扩散步数、网络宽度等），
    并实现 create() 返回一个可用的 BaseModel 实例。）
    """

    # 动作空间的维度（每个时间步动作向量的长度）。
    action_dim: int
    # 动作序列长度（一次预测多少步未来的动作）。
    action_horizon: int
    # tokenized prompt 的最大长度（用于 padding / 固定序列长度）。
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type. 返回模型类型（pi0 / pi0_fast / pi05），由子类实现。"""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters. 新建一个参数随机初始化的模型实例。rng 用于参数初始化。"""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters. 用给定参数创建一个模型（只换参数，不训练）。

        常用于把预训练 / 微调好的 checkpoint 参数装载进模型结构，供继续训练或推理。

        Args:
            params: 参数 PyTree（嵌套 dict，键为参数路径）。
            remove_extra_params: True 时只保留 params 与模型结构“共有”的参数，多余的键会被剔除，
                方便加载不同训练阶段 / 版本存下来的 checkpoint。
        """
        # eval_shape 只做“形状推导”，不真正分配 / 计算数组，
        # 从而可以零成本地得到一个只含形状信息的模型实例，用于拿到它的参数状态结构。
        model = nnx.eval_shape(self.create, jax.random.key(0))
        # NNX 把模型拆成两部分：
        #   graphdef = 纯网络结构（层拓扑，可哈希 / 可序列化）；
        #   state    = 所有参数 / 状态组成的 nnx.State（一个内部键带 "value" 后缀的 dict）。
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            # 取“交集”：只保留两边都有的参数键，剔除 checkpoint 里多出来的键。
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        # 结构校验：确保参数结构与模型状态完全一致（形状逐项比对，dtype 不比对）。
        # 不一致时 at.check_pytree_equality 会给出友好的报错信息。
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        # 用外来的 params 覆盖模型内部状态。
        state.replace_by_pure_dict(params)
        # 把“结构 + 参数”合并回一个完整可用的模型。
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        """加载 PyTorch 版模型权重（safetensors 格式）。

        推理 / 部署阶段常把模型转成 PyTorch 实现（见 openpi/models_pytorch/pi0_pytorch.py）。
        本方法先用 train_config 创建 PyTorch 模型骨架，再从 weight_path 读取权重填进去。
        """
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)  # 从 safetensors 文件原地加载权重到 model
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct.
        （返回模型输入的形状规格。

        返回值中的每个数组都是 jax.ShapeDtypeStruct——只有 shape 和 dtype、没有实际数值，
        用于给 XLA 编译器做静态形状推导、模型 jit 预热等场景。）
        """

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """构造一个“全是 1”的假观测，形状 / 类型与真实输入一致。

        用途：在正式跑数据之前，先用假数据把模型 jit 编译预热好，避免第一个 batch 卡顿。
        """
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        """构造一个“全是 1”的假动作，形状 / 类型与真实输出一致（常配合 fake_obs 一起使用）。"""
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).

    （所有模型实现的基类。具体模型继承本类，并在自己的 __init__ 里调用 super().__init__()
    以初始化共享属性 action_dim / action_horizon / max_token_len。）
    """

    # 这三个字段由配置（BaseModelConfig）传入，在训练 / 推理期间保持不变。
    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,  # 随机数 key（训练时扩散 / 去噪等过程需要采样噪声）
        observation: Observation,  # 观测输入
        actions: Actions,  # 真实动作（标签）
        *,
        train: bool = False,  # 是否训练模式（影响是否计算 EMA 等训练专用逻辑）
    ) -> at.Float[at.Array, "*b ah"]:
        """计算每个样本、每个动作步的损失，返回形状 [*b, ah]（批量 × 动作步数）。

        注意：这里返回的是“逐元素损失”，是否对 batch / horizon 求平均由训练循环决定。
        """

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        """给定观测，采样出要执行的动作（推理阶段使用）。

        不同模型实现方式不同：
        - pi0：基于扩散 / 流匹配，从噪声反复去噪得到动作轨迹；
        - pi0_fast：自回归地逐个 token 生成动作。
        kwargs 可传入推理相关参数（如扩散采样步数、温度等）。
        """


def restore_params(
    params_path: pathlib.Path | str,  # checkpoint 目录路径（支持本地路径或 gs:// 云盘路径）
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,  # 恢复成 jax.Array 还是 numpy 数组
    dtype: jnp.dtype | None = None,  # 统一恢复成该 dtype；None 表示保留 checkpoint 里的原始 dtype
    sharding: jax.sharding.Sharding | None = None,  # 参数的分片方式（多卡训练用）；None 则全设备复制
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint. 从 checkpoint 恢复“非结构化”的参数 PyTree。

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    （兼容两类 checkpoint：
    1. openpi 训练过程中由 save_state 保存的（见 training/checkpoints.py）；
    2. openpi 对外发布的预训练权重。）

    Args:
        params_path: The local path to the checkpoint directory. checkpoint 目录的本地路径。
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
            参数恢复成的数组类型。传 np.ndarray 可把参数全部加载为 numpy 数组。
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
            统一恢复的 dtype；不传则保留 checkpoint 原始 dtype。
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.
            参数的分片策略；不传则在所有设备上复制同一份。

    Returns:
        The restored params. 恢复好的参数 PyTree（嵌套 dict）。
    """
    # gs:// 前缀是 GCS 云存储路径，不能走 pathlib；本地路径则解析成绝对路径。
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 如果要以 jax.Array 恢复且未指定分片，默认在所有设备上复制：
    # 用 1 维 mesh + 空的 PartitionSpec 表示“整份参数在每个设备上各放一份”。
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:  # orbax 的 PyTree checkpoint 读取器（自动管理元数据缓存）
        metadata = ckptr.metadata(params_path)  # 读取 checkpoint 元信息（这里拿参数结构）
        item = {"params": metadata["params"]}  # 只关心 "params" 这一项（checkpoint 里可能还有 optimizer 状态等）

        # 真正执行恢复：按 metadata 里的结构，用统一的 restore_args 恢复每个数组。
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]  # 取出恢复结果中的 "params" 项

    # 兼容性处理：openpi 训练用 save_state 保存的参数，每个叶子键路径最后都带一个 "value" 后缀
    # （这是 nnx.State 序列化时引入的）。这里把后缀去掉，统一返回 NNX 所说的“pure dict”（干净的嵌套 dict）。
    flat_params = traverse_util.flatten_dict(params)  # 把嵌套 dict 拍平成 {(路径...): 值}
    if all(kp[-1] == "value" for kp in flat_params):  # 若所有叶子路径都以 "value" 结尾
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}  # 去掉最后一个 "value"
    return traverse_util.unflatten_dict(flat_params)  # 再还原成嵌套 dict
