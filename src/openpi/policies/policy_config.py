"""从训练好的 checkpoint 组装一个可以直接推理的 Policy（策略）对象。

一句话理解
----------
训练保存下来的 checkpoint 只是“裸权重”；要让机器人真正用起来，还需要：加载模型、
接上输入 / 输出变换（预处理与后处理）、准备归一化统计量（norm stats）。
本文件只提供一个函数 `create_trained_policy`，把上面这些步骤一次性做完，
返回 `openpi.policies.policy.Policy`——调用它的 `infer(obs)` 即可得到动作。

谁在用这个函数
--------------
- scripts/serve_policy.py：启动 WebSocket 服务前用它创建 policy；
- examples/ 与单元测试：直接调用它做离线推理 / 快速试跑。

checkpoint 目录长什么样
-----------------------
一个可用的 checkpoint 目录通常包含：

    <checkpoint_dir>/
      ├── params/                            # JAX(NNX) 权重，orbax 格式（目录）
      │                                      #   存在它 → train_config.model.load()
      ├── model.safetensors                  # PyTorch 权重（单文件）
                                             #   存在它 → train_config.model.load_pytorch()
      └── assets/<asset_id>/norm_stats.json  # 训练时保存的归一化统计量

`is_pytorch` 的判断就是“看目录里有没有 model.safetensors”。

一次 infer 的数据流（Policy 内部）
----------------------------------
输入（机器人观测 dict）：
    repack_transforms.inputs → InjectDefaultPrompt → data_transforms.inputs
    → Normalize → model_transforms.inputs → model.sample_actions
输出（动作 dict）：
    model_transforms.outputs → Unnormalize → data_transforms.outputs
    → repack_transforms.outputs

注意：输出侧的顺序与输入侧正好相反——模型输出要先逆掉“模型变换”，
再逆掉归一化，最后还原成机器人 / 数据集的字段格式。

新手阅读顺序
------------
1. 先看函数签名与 docstring：每个参数分别控制哪一步；
2. 再看函数体前几行：定位 / 下载 checkpoint，判断 JAX 还是 PyTorch；
3. 然后看模型加载与 norm stats 的来源；
4. 最后看 return 处的两个 transform 列表——它们定义了完整的输入 / 输出流水线。
"""

import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    从训练好的 checkpoint 组装出一个可直接推理的 Policy。

    Args:
        train_config: The training config to use to create the model.
            训练配置：既描述模型结构（train_config.model），也描述数据变换
            （train_config.data）和 policy 元信息（train_config.policy_metadata）。
        checkpoint_dir: The directory to load the model from.
            权重目录，支持本地路径或 gs:// 云路径（后者会自动下载到本地缓存）。
        repack_transforms: Optional transforms that will be applied before any other transforms.
            可选的“字段重打包”变换，在所有变换中最先执行。推理时一般不需要
            （客户端字段名通常已统一）；当输入用的是数据集字段名
            （如 observation.images.top）时才需要。
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
            传给底层 model.sample_actions 的额外参数（如采样步数、noise 等）；
            None 表示使用模型默认值。
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
            默认任务指令：当输入数据里没有 prompt 时自动注入（已有则不改）。
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
            归一化统计量。不传就从 checkpoint 目录的 assets/<asset_id>/norm_stats.json
            加载，以保证推理使用的统计量与训练时完全一致。
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
            PyTorch 模型运行的设备；JAX 模型会忽略该参数。

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
        通过检查 checkpoint 目录里是否存在 model.safetensors 来判断模型是否为 PyTorch 版。
    """
    # 不传 repack 时用空的 Group（inputs / outputs 都是空列表），
    # 这样后面写 `*repack_transforms.inputs` 时不用再判空。
    repack_transforms = repack_transforms or transforms.Group()
    # maybe_download：本地路径解析成绝对路径；gs:// 等远程路径先下载到本地缓存，
    # 返回一个保证存在的 pathlib.Path。下面的路径拼接都基于它。
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    # JAX checkpoint 存的是 params/ 目录（orbax 格式），
    # PyTorch 导出的是单个 model.safetensors 文件；用后者作为判别依据。
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        # PyTorch 路径：先按 train_config 建出 PI0Pytorch 骨架，再把 safetensors 权重填进去；
        # 随后把部分参数转成 bfloat16，兼顾数值稳定与显存占用
        # （与训练 / 导出时的设定保持一致）。
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # JAX 路径：
        # 1. restore_params 从 checkpoint_dir/params 读出参数 PyTree，并统一成 bfloat16；
        # 2. train_config.model.load(...) 按相同的模型结构“套上”这些参数，返回可推理的模型。
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    # 用训练配置里的“数据工厂”实例化出 DataConfig：
    # 里面有 data_transforms / model_transforms / asset_id / use_quantile_norm 等，
    # 正是下面组装输入输出流水线所需要的信息。
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        # 刻意从 checkpoint 的 assets/ 加载，而不是从训练配置的 assets 目录加载：
        # 保证“这次推理用的归一化统计量”与“当初训练这份权重时用的”完全一致。
        if data_config.asset_id is None:
            # asset_id 是 assets/ 下的子目录名（通常等于数据集 repo_id），
            # 没有它就无法定位 norm_stats.json。
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    # PyTorch 模型需要显式指定设备：用户没给就自动选（有 GPU 用 cuda，否则 cpu）。
    # JAX 模型的设备由 JAX 自己管理，因此不受这里影响。
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # 环境里没装 PyTorch？退化为 cpu（后续加载时自然会报出更明确的错误）。
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            # ---- 输入流水线（按列表顺序依次执行）----
            # 1) 可选的字段重打包：把外部字段名翻译成 openpi 统一字段名；
            *repack_transforms.inputs,
            # 2) 输入里没有 prompt 时注入默认任务指令；
            transforms.InjectDefaultPrompt(default_prompt),
            # 3) 机器人 / 数据集特有的变换（坐标系、夹爪约定等）；
            *data_config.data_transforms.inputs,
            # 4) 归一化：让输入分布与训练时一致（分位数或 z-score）；
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 5) 模型特有变换：resize 图像、分词 prompt、补齐维度等。
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            # ---- 输出流水线（按列表顺序依次执行，整体是输入流程的逆序）----
            # 1) 模型输出的逆变换：把动作从模型内部格式还原回来；
            *data_config.model_transforms.outputs,
            # 2) 反归一化：把归一化空间的动作还原成真实的物理量纲；
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 3) 数据侧逆变换（恢复到机器人 / 数据集约定的动作表示）；
            *data_config.data_transforms.outputs,
            # 4) 可选的字段重打包（恢复成调用方期望的字段名）。
            *repack_transforms.outputs,
        ],
        # sample_kwargs=采样参数（None 时 Policy 内部用空 dict，即模型默认行为）；
        # metadata 会随连接发给客户端（例如 WebSocket 服务器第一帧发送它），
        # 因此里面的值必须能被 msgpack 序列化（如 list / dict / 数字，而不是自定义对象）。
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,  # JAX 模型不需要设备参数
    )
