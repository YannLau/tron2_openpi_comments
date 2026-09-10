"""预训练权重加载器：训练启动时，如何把已有的外部权重“灌入”刚初始化好的模型。

一句话理解
----------
训练大模型通常不是每次从零开始：要么接着别人训练好的 checkpoint 继续训练 / 微调，
要么先用一个公开的基座权重初始化主干。本文件把“如何初始化”抽象成若干个可配置的
加载器（WeightLoader），由 `TrainConfig.weight_loader` 字段引用（见 training/config.py）。

调用发生在哪
------------
- 训练脚本 scripts/train.py 的 init_train_state 中，在“非断点续训”的情况下执行
  `loader.load(params)`：此时 `params` 是模型刚随机初始化出来的参数树
  （该路径上通常只是 jax.eval_shape 得到的形状占位符：只有 shape/dtype，不含真实数值）；
- 加载器返回“checkpoint 里应采用的键 + 模型自身应保留的键”，训练脚本随后把返回的
  真实数组填回模型（见 scripts/train.py 的 _load_weights_and_validate）。

接口约定（WeightLoader.load 的输入输出）
----------------------------------------
1. 输入 `params` 与返回值必须是同一个嵌套结构（参数树）；
2. 若 checkpoint 只覆盖一部分参数（例如只有主干、没有 LoRA），加载器仍需要用
   `params` 里的默认值补齐缺失键，保证返回结构完整；
3. 加载器只负责“取权重 + 合并”，dtype 转换、转成 jax.Array、多卡分片等后续步骤
   统一由训练代码处理。

文件结构（新手阅读顺序）
------------------------
1. `WeightLoader`（协议）：先看接口长什么样；
2. `NoOpWeightLoader` / `CheckpointWeightLoader` / `PaliGemmaWeightLoader`：三种策略，
   每个实现都只有“取外部权重 → 调 _merge_params 合并”两步；
3. `_merge_params`（本文件核心算法）：理解“按路径字符串匹配，覆盖 / 保留 / 丢弃”。

两个小知识点
------------
- 模型参数是“嵌套 dict”形式的 PyTree（叶子为张量）。flax.traverse_util.flatten_dict
  能把嵌套 dict 拍平成 {"父/子/孙": 张量} 的扁平 dict，方便用字符串 / 正则做键匹配；
- unflatten_dict 是其逆操作，把扁平 dict 还原成嵌套 dict。
"""

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    """加载器接口：所有 weight loader 都必须实现 `load(params)` 方法。

    用 Protocol 而不是普通基类，意味着“鸭子类型”——任何实现了 `load` 的对象都可以
    当作加载器传入配置，不必强制继承本类。@runtime_checkable 额外允许在运行时用
    `isinstance(obj, WeightLoader)` 检查（仅靠 Protocol 时这种检查只服务于静态类型工具）。
    """

    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        加载 / 合并外部权重，并返回与 `params` 结构一致的参数树。

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.
                当前模型的参数树（嵌套结构，叶子是类数组对象）。
                训练启动流程中传入的多是只含 shape/dtype 的形状占位符，
                加载器负责把真实权重放到对应位置。

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
            合并后的参数树。返回结构必须与 `params` 完全一致；如果只加载部分参数，
            加载器必须用 `params` 中的默认值补齐其余键，保证调用方拿到的是完整结构。
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    """“空操作”加载器：什么都不加载，原样返回参数。

    使用场景：从零开始训练（不想要任何预训练权重）时。因此它也是
    `TrainConfig.weight_loader` 的默认值（见 training/config.py 中的 dataclass 字段）。

    frozen=True 表示实例不可变（纯配置对象）；本类没有任何字段，所以
    所有 NoOpWeightLoader 实例的行为完全一致。
    """

    def load(self, params: at.Params) -> at.Params:
        # 原样返回：模型参数保持刚被随机初始化（或调用方传入）时的状态，
        # 不做任何覆盖。
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    从一个 checkpoint 加载整套权重——最常用的加载器。

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    兼容两类来源：自训 checkpoint 目录（训练中由 save_state 落盘，见
    training/checkpoints.py），以及 openpi 官方发布的预训练权重目录（gs:// 云存储路径）。

    加载策略一句话：
    - checkpoint 里有的键 → 覆盖模型对应位置的随机初始化值；
    - checkpoint 里没有、但路径含 “lora” 的键（如微调时才新增的 LoRA 适配器）
      → 保留模型自己的初始值，从而保证返回结构与当前模型参数完全一致。
    """

    # checkpoint 路径：本地目录或 gs:// 远程目录（远程下载由 maybe_download 处理）。
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # maybe_download：本地路径直接解析为绝对路径；
        # gs:// 路径先下载到本地缓存再返回 Path。
        # restore_params：用 orbax 读取 checkpoint 目录，恢复成嵌套 dict 形式的参数树。
        # restore_type=np.ndarray 表示先加载成普通 NumPy 数组——这一步还在宿主 CPU 侧，
        # dtype 转换、转成 jax.Array、FSDP 分片等统一交给后续训练代码
        # （scripts/train.py 与 training/sharding.py）。
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # 官方发布的基础 checkpoint 通常不含 LoRA 权重（LoRA 是微调时才加上的适配器），
        # 而自训 checkpoint 可能已经包含。missing_regex=".*lora.*" 的含义是：
        # 凡 checkpoint 没提供、且路径含 lora 的键，都从当前 params（新初始化的 LoRA）
        # 补齐；其余缺失键不被补，若真的缺失会在调用方的结构校验中报错。
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.

    用 Google 官方 PaliGemma checkpoint 初始化模型的 VLM（视觉语言）主干部分。中文解读：
    - 官方 checkpoint 提供的是纯视觉语言模型权重（SigLIP 图像编码器 + Gemma 语言模型）；
    - “覆盖同名权重”：官方权重里凡与当前模型路径相同的键（即 PaliGemma 主干部分）
      都会被官方值替换；
    - “保留额外权重”：当前模型多出来的键（pi0 的动作专家、各类投影层、LoRA 等）
      不会被删除，继续使用模型刚初始化时的值；
    - 这正是用“纯 VLM 权重”起步去训练 pi0 这类“VLM + 动作专家”组合模型的关键设计。

    注意：下载 URL 写死在 Google Cloud Storage 上，并以匿名公开方式读取；
    无外网环境会失败。
    """

    def load(self, params: at.Params) -> at.Params:
        # 下载官方 .npz 权重文件（pt_224 表示输入分辨率 224 的 PaliGemma 变体）。
        # gs={"token": "anon"} 是传给底层 fsspec 的参数：匿名访问 GCS 公开对象，无需鉴权。
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        # .npz 是 NumPy 的打包格式：np.load 得到 {"扁平键": 数组} 的字典，
        # 键形如 "params/llm/..." 或 "params/img/..."（用 / 表示层级）。
        # allow_pickle=False 只允许纯数组数据，
        # 避免执行文件里可能嵌入的 pickle 代码（安全考虑）。
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        # 三步还原成与当前模型匹配的结构：
        # 1. unflatten_dict 把 "a/b/c" 形式的扁平键还原成嵌套 dict；
        # 2. 官方文件在最外层包了一个 "params" 命名空间，这里把它去掉；
        # 3. 再包一层 {"PaliGemma": ...}，对齐本仓库模型的顶层参数名
        #    （pi0 参数树顶层形如 PaliGemma/...、action_in_proj/... 等）。
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # missing_regex=".*" = 所有未被官方权重覆盖的键都保留模型自身的值，
        # 这样合并结果的结构与当前模型参数完全一致（这是接口 / 结构校验的要求）。
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    把“外部 checkpoint 参数”与“目标模型参数”按路径合并；
    这是所有加载器共用的核心算法。

    合并规则（对应下方代码的两步）：
    1. 覆盖：checkpoint 中与目标模型路径相同的键 → 采用 checkpoint 的值
       （dtype 不一致时按目标模型的 dtype 转换）；
    2. 丢弃：checkpoint 中目标模型没有的“多余键” → 忽略，不写入结果；
    3. 保留：目标模型里未被覆盖、且完整路径匹配 missing_regex 的键 → 沿用目标模型
       自己的值（典型例子：checkpoint 里没有的 LoRA / 新增模块）。

    Args:
        loaded_params: The parameters to merge. 外部来源（checkpoint）加载出的参数树。
        params: The reference parameters.
            目标模型的参数树（既是“参考结构”，也是缺失键的“默认值”来源）。
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
            需要从目标模型补齐的“缺失键”路径正则（用 re.fullmatch 做整串匹配，
            例如 ".*lora.*" 只补齐路径含 lora 的键，".*" 补齐所有键）。

    Returns:
        A new dictionary with the merged parameters. 合并后的新参数树（不修改两个入参）。
    """
    # 把两层嵌套 dict 都拍平成 “路径字符串 → 张量” 的扁平 dict。
    # sep="/" 使路径风格与 checkpoint / npz 的命名一致，也便于后续用正则匹配整条路径。
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # ---- 第 1 步：先取“checkpoint 中属于目标模型的那部分键”，用它们覆盖随机初始化 ----
    # 只有 k 同时存在于 flat_ref 才写入 result：checkpoint 里多出来的键
    # （例如旧版本遗留的参数）在目标模型中不存在，直接丢弃。
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            # dtype 以目标模型为准：checkpoint 里常见 float32，而模型可能是 bf16，
            # 这里做一次显式转换（astype 会生成新数组，不影响 checkpoint 源数据）。
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    # 清空扁平字典，尽快释放未被采用的键的引用。
    # 注意：已写进 result 的数组仍被 result 引用，不会因此被误删。
    flat_loaded.clear()

    # ---- 第 2 步：把“checkpoint 没覆盖、但符合 missing_regex 的目标键”补回来 ----
    # re.fullmatch 要求整条路径匹配（不是子串搜索）：
    #   missing_regex=".*lora.*" → 只保留路径含 lora 的目标键；
    #   missing_regex=".*"       → 保留所有未被覆盖的目标键。
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:  # 已被第 1 步覆盖的键不重复处理
            result[k] = flat_ref[k]  # 沿用目标模型自身的参数（默认值 / 形状占位符）

    # 把扁平 dict 还原成与 params 相同的嵌套 PyTree 结构并返回。
    return flax.traverse_util.unflatten_dict(result, sep="/")
