"""数据变换（transforms）：在“原始数据”和“模型输入 / 输出”之间做结构、字段与量纲的转换。

一句话理解
----------
模型只认识一种固定的数据格式：图像是 [-1, 1] 的 float、state / actions 是固定维度的数组、
prompt 已经分词成 token……而数据集 / 机器人送来的数据五花八门。本文件定义了各种“变换”
（transform）：每个变换就是一个可调用的对象，接收一个字典，返回变换后的字典；
把它们按顺序串起来，就构成了训练 / 推理的数据流水线。

一个变换长什么样
----------------
变换的输入输出都是“嵌套 dict”（PyTree），叶子通常是 NumPy 数组，常用键有：

    image            : {相机名: 图像数组}，NHWC 布局（batch, height, width, channel）
    image_mask       : {相机名: bool}，标记该相机数据是否有效
    state            : 机器人状态（关节角等），形状 [..., state_dim]
    actions          : 动作序列，形状 [..., action_horizon, action_dim]
    prompt           : 任务指令（字符串或 0 维 numpy 字符串数组）
    tokenized_prompt : 分词后的 prompt token 序列
    tokenized_prompt_mask : 与上面配套的 mask（区分真实 token 与 padding）

变换在整个流水线中的位置（Policy 内部，组装逻辑见 policies/policy_config.py）：

    输入：原始观测 → repack → InjectDefaultPrompt → data_transforms → Normalize
          → model_transforms → model.sample_actions
    输出：模型输出 → model_transforms.outputs → Unnormalize
          → data_transforms.outputs → repack.outputs

本文件内容分类（也就是建议的阅读顺序）
--------------------------------------
1. 组合机制：DataTransformFn（接口）→ Group（一组输入 / 输出变换）→ CompositeTransform / compose；
2. 结构重排：RepackTransform、transform_dict、flatten_dict / unflatten_dict；
3. prompt 相关：InjectDefaultPrompt、PromptFromLeRobotTask、TokenizePrompt、TokenizeFASTInputs；
4. 数值归一化：Normalize / Unnormalize（配合 apply_tree、pad_to_dim、_assert_quantile_stats）；
5. 动作空间与形状：DeltaActions / AbsoluteActions（配合 make_bool_mask）、SubsampleActions、
   PadStatesAndActions、ResizeImages；
6. pi0-FAST 专用：ExtractFASTActions（把模型输出的离散 token 解码回连续动作）。

几个容易踩的坑（写在前面）
--------------------------
- 部分变换会“原地修改”传入的数组（例如 DeltaActions 直接对 actions 做 -=）；
  如果调用方还需要原始数据，请先自行复制。
- 返回值可能是传入的同一个对象，也可能是新字典（例如用 {**data, ...} 构造的）；
  调用方统一使用返回值即可，不要假设数据一定被原地修改。
- 在 DataLoader 的 worker 进程里请用 NumPy 而不是 JAX 数组，避免无谓占用 GPU 显存。
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

# 变换的输入 / 输出类型：一个 PyTree（嵌套 dict，叶子是数组）。
DataDict: TypeAlias = at.PyTree
# 一组归一化统计量（mean / std / q01 / q99），定义见 shared/normalize.py。
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    """变换的“接口”（Protocol）：任何可调用对象，只要 `__call__(data) -> data` 就算一个变换。

    使用 Protocol 的好处是：既可以写类（如本文件里的各种 dataclass），
    也可以直接传一个普通函数或 lambda，框架都能接受。
    @runtime_checkable 允许运行时用 isinstance(obj, DataTransformFn) 检查。
    """

    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        对数据施加变换。

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.
                待变换的数据：可能是嵌套 dict，元素是“没有 batch 维度”的单条数据，
                叶子应当是 NumPy 数组（用 JAX 数组也可以，但 DataLoader worker 里不推荐，
                因为可能额外占用 GPU 显存）。

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
            变换后的数据。既可能是被原地修改的 `data`，也可能是一个新的数据结构。
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms. 把若干输入变换与输出变换打成一个包。

    为什么要成对存放？因为输出流水线必须是输入流水线的“逆序”：
    例如输入侧先做 DeltaActions 再做 Normalize，输出侧就要先 Unnormalize 再 AbsoluteActions。
    训练 / 推理框架（见 policies/policy_config.py）只需要一个 Group，就能同时拿到两侧的变换。

    `frozen=True` 表示不可变：push() 不会改原对象，而是返回新的 Group。
    """

    # Transforms that are applied to the model input data.
    # 作用在“模型输入”上的变换（按列表顺序依次执行）。
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    # 作用在“模型输出”上的变换（按列表顺序依次执行，通常与 inputs 顺序相反）。
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        追加变换并返回一个新的 Group。

        Args:
            inputs: Appended to the *end* of the current input transforms.
                追加到现有输入变换的“末尾”（流水线最后执行）。
            outputs: Appended to the *beginning* of the current output transforms.
                插入到现有输出变换的“最前面”（最先执行）。
                这样安排是为了让输出顺序自动与输入顺序保持对称的逆序关系。

        Returns:
            A new group with the appended transforms.
            包含新变换的 Group（原 Group 不变）。
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order.

    把一串变换按顺序组合成单个变换。
    """

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)  # 前一个变换的输出作为后一个变换的输入
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    # 便捷函数：Policy 内部就是用它把 transform 列表变成一个可调用对象。
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }

    把输入字典“重排 / 改名”成新的字典结构：结构（structure）的键是新名字，
    值是旧数据里用 '/' 连接的路径。例如上面的例子会把：

        {"observation": {"images": {"top": img}, "state": s}, "action": a}

    变成：

        {"images": {"cam_high": img}, "state": s, "actions": a}

    这个变换通常放在流水线最前面（data_transforms 之前），
    用来把“数据集自己的字段名”翻译成 openpi 的统一字段名。
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        # 先把输入拍平成 {"observation/images/top": 数组, ...}，方便按路径取值。
        flat_item = flatten_dict(data)
        # structure 同样是一个嵌套 dict，其叶子是“旧路径”字符串；
        # jax.tree.map 会保持 structure 的嵌套形状，把每个叶子替换成 flat_item 里的数组。
        # 注意：这里只重新组织引用，不会复制数组数据。
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    """当数据里没有 prompt 时，注入一个默认任务指令。

    prompt 是给 VLM 的语言指令（例如 “pick up the cup”）。推理时客户端可能不传 prompt，
    这时就用这里配置的默认值；若数据里已经有 prompt，则保持原样、不覆盖。
    """

    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            # 包成 0 维 numpy 字符串数组，与其它数据 leaf 的“数组”风格保持一致；
            # 后续 TokenizePrompt 会用 prompt.item() 取出 Python 字符串。
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    """把数据归一化到模型友好的数值范围（输入侧使用）。

    两种模式（由 use_quantiles 决定）：
    - z-score（默认）：(x - mean) / std，把数据变成均值 0、标准差 1；
    - quantile：用 1% / 99% 分位数把数据线性映射到 [-1, 1]，对异常值更稳健。
      pi0.5 / pi0-FAST 采用这种模式（由 DataConfig.use_quantile_norm 决定）。

    `norm_stats` 是一棵与数据 key 对应的统计量树（例如 state / actions 各一份），
    结构与数据一致；apply_tree 会按路径把两者匹配起来。
    """

    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    # True 用分位数归一化；False 用普通 z-score 归一化。
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    # True 时，如果统计量里有某个 key 而数据里没有，就报错（用于尽早发现配置不匹配）。
    strict: bool = False

    def __post_init__(self):
        # dataclass 构造完成后立刻做一次校验：分位数模式必须有 q01 / q99，
        # 否则等到使用时才报错会很难定位。
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        # norm_stats=None 表示“本数据集不做归一化”，直接放行。
        if self.norm_stats is None:
            return data

        # 按扁平化的 key 匹配：数据里的每个叶子，如果在统计量树里有同路径的统计量，
        # 就调用 _normalize / _normalize_quantile 处理；其余叶子原样保留。
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        # 只取前 x.shape[-1] 个维度的统计量（[..., : x.shape[-1]]）：
        # 动作维度可能被 PadStatesAndActions 补过零，统计量维度不一定完全相等。
        # +1e-6 是为了防止极小的 std 造成除零。
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        # 分位数归一化：q01 → -1，q99 → +1（线性映射）；
        # 超出范围的值会被放大到 [-1, 1] 之外。
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    """把模型输出从归一化空间还原回真实物理量纲（输出侧使用）。

    是 Normalize 的逆变换，必须与输入侧使用同一份 `norm_stats`，
    并且调用顺序上要“外一层”执行：输入是 Normalize 在前、Unnormalize 在后，
    输出则是 Unnormalize 在前、真正的数据逆变换在后（见 policy_config.py 的组装代码）。
    """

    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    # True 用分位数归一化；False 用普通 z-score 归一化。必须与输入侧的 Normalize 保持一致。
    use_quantiles: bool = False

    def __post_init__(self):
        # 与 Normalize 相同：分位数模式要求统计量里必须有 q01 / q99。
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        # 这里固定 strict=True：模型输出的每个动作 key 都必须能反归一化，
        # 缺少统计量说明 norm_stats 与模型 / 配置不匹配，应当立刻报错而不是悄悄跳过。
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        # z-score 的逆运算：x * std + mean。
        # 用 pad_to_dim 把 mean 补成 0、std 补成 1，这样“多出来的填充维度”保持不变
        # （乘 1 加 0 = 原值），不会影响其它维度的数值。
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        # 分位数归一化的逆运算：-1 → q01，+1 → q99。
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            # 统计量只覆盖前 dim 维（例如机器人只有 7 维动作，但模型动作维是 32）：
            # 只逆变换前 dim 维，后面多出来的填充维度原样保留（用 concatenate 拼回去）。
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    """把 `data["image"]` 里的每相机图像缩放到指定分辨率。

    使用 resize_with_pad：等比缩放后再补零，保证不拉伸变形；
    模型（PaliGemma）要求固定输入尺寸，所以这是 model_transforms 里的常见一步。
    """

    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        # data["image"] 是 {相机名: 图像数组}；对每个相机单独缩放后放回原字典。
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    """对动作序列做下采样：每隔 stride 步保留一个动作。

    例如 stride=2 会把 50 步的 action chunk 变成 25 步。
    改变动作步数时必须同步调整模型配置里的 action_horizon，否则形状对不上。
    """

    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        # Python 切片 [::stride] 作用在最外层（时间）维度上。
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space.

    把“绝对动作”转换成“相对当前状态的增量动作”：对 mask 为 True 的维度计算
    `actions - state`，mask 为 False 的维度保持绝对值不变。

    典型用法：双臂机器人的 6 个关节角用增量动作（更容易学习），夹爪维度保持绝对值。
    输入侧用本变换，输出侧必须配对的 `AbsoluteActions`（用同一 mask）还原成绝对动作。
    """

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    # 布尔掩码：True 的维度转成增量，False 的维度保持绝对值；
    # 长度可以小于实际动作维度（只影响前若干维）；None 表示整个变换不生效（no-op）。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # actions 形状 [..., horizon, dim]，state 形状 [..., dim]；
        # np.where(mask, state, 0) 先把不该转增量的维度置 0，
        # expand_dims(..., axis=-2) 再扩成 [..., 1, dim]，从而能广播减到每个时间步上。
        # 注意：这里是对 actions 数组原地减法（-=），会修改传入的数据。
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space.

    把“增量动作”还原回“绝对动作”：对 mask 为 True 的维度计算 `actions + state`。
    它是 `DeltaActions` 的逆操作，通常放在输出侧
    （模型采样出的动作经过反归一化后再执行本变换）。
    必须与输入侧的 DeltaActions 使用完全相同的 mask。
    """

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    # 布尔掩码：True 的维度加回 state，False 的维度保持原样；
    # 长度可以小于实际动作维度；None 表示整个变换不生效（no-op）。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # 与 DeltaActions 对称：把 state 广播到每个时间步后加回去（原地修改）。
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    """把 prompt（pi0.5 还可选择把 state）编码成 token 序列。

    产出 `tokenized_prompt` 与 `tokenized_prompt_mask` 两个字段供模型使用；
    prompt 本身会从数据里移除（pop），避免后续步骤重复处理。
    输入的 prompt 必须存在，否则直接报错（可用 InjectDefaultPrompt 兜底）。
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    # pi0.5 支持“离散状态输入”：把 state 也作为文本 token 一起编码。
    # True 时必须提供 state，否则报错。
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        # pop：取出 prompt 并从 data 中删掉；取不到（None）说明流水线配置缺少 prompt 来源。
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        # prompt 可能是 0 维 numpy 字符串数组（InjectDefaultPrompt 就是这么放的），
        # 用 .item() 取出真正的 Python str 再交给分词器。
        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        # {**data, ...}：保留其余字段，仅追加分词结果（原 prompt 已被 pop 掉）。
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    """pi0-FAST 训练专用的输入分词：把 prompt + state + actions 一起编码成 token。

    与 TokenizePrompt 的区别：
    - pi0-FAST 把动作也离散化（分箱）成 token，用自回归方式生成；
    - 因此除 prompt 的 token / mask 外，还产出 `token_ar_mask`（自回归注意力 mask）
      与 `token_loss_mask`（哪些位置参与损失计算，即 teacher forcing 的目标位置）。
    推理时数据里通常没有 actions，分词器会按“只编码输入”的方式处理。
    """

    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        # state 必须有；actions 可选（训练时有、推理时无），所以用 .get()。
        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    """pi0-FAST 推理专用的输出解码：把离散 token 还原成连续动作。

    FAST 模型生成的 `actions` 其实是动作 token 的编号；本变换调用分词器的
    `extract_actions`，把它们解码成形状 [action_horizon, action_dim] 的连续动作，
    这样后续的 Unnormalize / AbsoluteActions 等流程就能和 pi0 完全共用。
    """

    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        # 先 pop 掉 token 形式的 "actions"，再在同一位置放回解码后的连续动作。
        tokens = data.pop("actions")
        # astype(np.int32)：模型输出可能是浮点，转成整数 token id 再解码。
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task.

    从 LeRobot 数据集的 task 映射表里取出当前样本对应的任务文本，写入 `prompt`。

    训练时数据样本通常只有一个 `task_index`（整数），而 `tasks` 是
    `dataset.meta.tasks`（{task_index: 任务文本}）。本变换负责把两者对应起来。
    """

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    # 形如 {0: "pick up the cup", 1: "open the drawer", ...}。
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        # 保留其余字段，仅追加（或覆盖）prompt。
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension.

    把 state 与 actions 的最后一维用 0 补齐到模型要求的维度（`model_action_dim`）。

    不同机器人的自由度不同（例如 7 关节 + 1 夹爪），而模型的输入 / 输出维度是固定的，
    所以统一补齐；补齐前不足的维度为 0，多余维度不会被截断（pad_to_dim 只在“不够”时补）。
    state 一定处理；actions 存在时才会处理（推理时可能没有 actions）。
    """

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            # 训练数据里通常带 actions；推理输入没有该字段。
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    # 例如 {"a": {"b": 1}} -> {"a/b": 1}；用 '/' 是为了和 RepackTransform /
    # transform_dict 里配置的路径字符串保持同一套写法。
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    # flatten_dict 的逆操作：{"a/b": 1} -> {"a": {"b": 1}}。
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.

    用一组“正则 → 新名字”的规则批量重命名 / 删除嵌套字典里的叶子。
    - patterns 的键是正则表达式（必须整条路径 fullmatch 才算匹配），
      值是替换后的新路径（可用 \\1 之类的反向引用），值为 None 表示删除该叶子；
    - 路径都以 '/' 分隔（内部先把树拍平）；
    - 规则按 dict 的插入顺序生效，只有“第一个匹配上的规则”会被使用；
    - 没被任何规则匹配的叶子保持原样。

    例子：
        transform_dict({"(.+)/c": r"\\1/d"}, {"a": {"b": 1, "c": 2}})
        # -> {"a": {"b": 1, "d": 2}}（所有 .../c 改名为 .../d）
        transform_dict({"a/b": None}, {"a": {"b": 1, "c": 2}})
        # -> {"a": {"c": 2}}（删除 a/b）
    """
    # 先拍平，后续所有匹配都作用在 "父/子/孙" 这样的完整路径字符串上。
    data = flatten_dict(tree)

    # Compile the patterns.
    # 预编译正则；普通 dict 在 Python 3.7+ 会保持插入顺序，
    # 所以“第一条匹配的规则生效”是确定的行为。
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            # 必须整条路径匹配（fullmatch）：pattern="a" 不会匹配到 "a/b"。
            if pattern.fullmatch(k):
                # repl 为 None 表示删除；否则做一次替换（count=1），支持反向引用。
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            # for-else：上面没有 break（所有规则都不匹配）时执行，保留原始 key。
            new_k = k

        if new_k is not None:
            if new_k in output:
                # 两条路径被重命名成同一个名字会导致数据被覆盖，直接报错。
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    # 合法性检查：不能同时存在 "a" 和 "a/b" 这种“叶子又当父节点”的情况，
    # 否则 unflatten 时会冲突（例如 {"a": 1, "a/b": 2} 无法还原成嵌套 dict）。
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    """按“路径”把 selector 里的值与 tree 对应叶子配对，并对每个配对的叶子调用 fn。

    可以理解为“树版的 dict.update 计算”：以 selector 的路径为准，
    只要 tree 里有同路径的叶子，就执行 fn(tree_leaf, selector_value)，
    其余叶子保持不变。Normalize / Unnormalize 就是用它把统计量套到数据上的。

    Args:
        tree: 要被处理的数据树（叶子为张量）。
        selector: 提供“每个路径对应的参数”的树（如 NormStats）。
        fn: 处理函数 fn(数据叶子, 参数叶子) -> 新数据叶子。
        strict: True 时，如果 selector 里的某个路径在 tree 中不存在，直接报错；
            False 时忽略这些路径。
    """
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        # 只有路径能在 selector 里找到时才施加 fn。
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        # 先整体校验：selector 中的每个 key 都必须在 tree 中存在。
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    # 逐叶子处理后还原成嵌套结构。
    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    # 沿指定轴把数组补齐到 target_dim（用 value 填充，默认 0）。
    # 已经够长时原样返回（不截断）；axis 可用负数表示“从后往前数”，如 -1 是最后一维。
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        # np.pad 需要为每个维度指定 (前补, 后补)；其余维度补 0，只有目标轴补差值。
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.

    按“正数写几个 True、负数写几个 False”的方式快速拼出布尔掩码。
    主要用来构造 DeltaActions / AbsoluteActions 的 mask：
    True 的维度做“绝对 ↔ 增量”转换，False 的维度保持绝对值。

    例：ALOHA 双臂每只手臂 6 个关节 + 1 个夹爪，希望关节转增量、夹爪保持绝对值：
        make_bool_mask(6, -1, 6, -1)
        # -> (T,T,T,T,T,T, F, T,T,T,T,T,T, F)
    """
    result = []
    for dim in dims:
        if dim > 0:
            # 正数 n：追加 n 个 True。
            result.extend([True] * (dim))
        else:
            # 负数 -n：追加 n 个 False；0 不追加任何元素。
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    """分位数归一化的前置校验：每个 NormStats 叶子都必须带 q01 / q99。

    否则等到真正使用时才因为缺字段崩溃，很难定位是哪个 key 出了问题；
    这里提前报错，并在错误信息里带上对应的 key 路径。
    """
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
