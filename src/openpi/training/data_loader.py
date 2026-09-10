"""数据加载器：把磁盘 / 云端的数据集变成训练循环可以直接消费的 batch 流。

一句话理解
----------
`scripts/train.py` 每走一步都要拿到一个 batch，本文件负责把这个 batch “造”出来：
读取原始数据 → 拼出动作序列 → 套用数据变换与归一化 → 打包成固定形状的批量数组，
并（在 JAX 下）按设备分好片。训练循环收到的是 `(Observation, Actions)` 元组，
其中 `Observation` 是 models/model.py 里定义的结构化观测类型。

数据流（从原始数据到训练输入）
------------------------------
    原始数据
      ├── LeRobot 数据集（随机访问）──▶ create_torch_dataset
      ├── RLDS / DROID 数据集（流式）─▶ create_rlds_dataset
      └── 假数据（debug / 单测）─────▶ FakeDataset
                                        │
                     transform_dataset / transform_iterable_dataset
                     （repack → data_transforms → Normalize → model_transforms）
                                        │
                  TorchDataLoader / RLDSDataLoader（batching、shuffle、多进程、分片）
                                        │
                          DataLoaderImpl.__iter__（打包成 (Observation, Actions)）
                                        │
                               训练循环 scripts/train.py

本文件的主要角色
----------------
- 协议接口：Dataset（随机访问）、IterableDataset（流式）、DataLoader（统一迭代入口）；
- 数据包装：TransformedDataset / IterableTransformedDataset（惰性地对每个样本施加变换）、
  FakeDataset（无需真实数据即可跑通流程）；
- 具体加载器：TorchDataLoader（基于 PyTorch DataLoader，支持多 worker）、
  RLDSDataLoader（RLDS 数据集自己已经批处理，这里只是薄包装）；
- 统一出口：DataLoaderImpl（把上面两者统一成“产出 (Observation, Actions)”）。

几个容易混淆的概念
------------------
- 全局 batch_size vs 本地 batch_size：`config.batch_size` 是所有设备加起来的批量；
  多进程 / DDP 时会先除以进程数（必要时再除以 world size）得到“本进程批量”；
- num_batches：None 表示“无限迭代”（数据集跑完自动从头再来），
  给定时表示最多产出多少个 batch（测试 / 统计归一化时常用）；
- 数据变换的实际实现都在 openpi/transforms.py，本文件只负责“按顺序接起来”。

新手阅读顺序
------------
1. 先看三个 Protocol（Dataset / IterableDataset / DataLoader），理解接口约定；
2. 再看 create_data_loader（总入口）与 create_torch_dataset（数据集怎么来）；
3. 然后看 transform_dataset / transform_iterable_dataset（变换链如何拼装）；
4. 最后看 TorchDataLoader（多 worker、分片、无限循环）与 DataLoaderImpl（最终出口）。
"""

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _load_lerobot_dataset_module():
    # Deployment-only temporary support: real-robot inference does not need
    # lerobot/av, so keep lerobot optional unless a LeRobot dataset is loaded.
    # 真机推理不需要 lerobot（及其 av 依赖），所以这里把导入推迟到“真的要用数据集”时再做：
    # - 好处：部署环境可以少装一大堆依赖；
    # - 代价：忘装依赖时错误只会在这里出现，因此错误信息里直接给出解决办法。
    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    except ImportError as exc:
        raise ImportError(
            "lerobot is required for LeRobot dataset loading. Uncomment the "
            "lerobot dependency in pyproject.toml and run `uv sync` before "
            "training or loading LeRobot datasets."
        ) from exc

    return lerobot_dataset


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access.

    支持“随机访问”的数据集接口：可以用下标取第 i 条样本，也知道总长度。
    LeRobot 数据集与本文件里的 FakeDataset 都属于这一类。
    """

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset.

    只能“顺序迭代”的数据集接口（流式数据）：不保证能随机访问，
    例如 RLDS / DROID 这种由 TensorFlow 流水线按顺序吐出的数据。
    """

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader.

        返回创建本加载器时使用的 DataConfig。checkpoint 保存流程会用它拿到
        norm_stats / asset_id 等信息，把归一化统计量一起存进 checkpoint。
        """
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    """给“随机访问数据集”套一层变换：每次 __getitem__ 时才对单条样本做变换。

    这样做的好处：
    - 惰性计算：只有在真正取到某条数据时才做变换，不会提前把整个数据集处理一遍；
    - 与 DataLoader 的多 worker 天然配合：每个 worker 进程各自处理自己取到的样本。
    """

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)  # 把变换列表组合成单个可调用对象

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])  # 先取原始样本，再依次施加所有变换

    def __len__(self) -> int:
        return len(self._dataset)  # 长度与底层数据集一致


class IterableTransformedDataset(IterableDataset[T_co]):
    """给“流式数据集”套一层变换。

    与 TransformedDataset 的区别在于数据是逐批 / 逐条“流”出来的；
    当底层数据集吐出的已经是 batch（`is_batched=True`，RLDS / DROID 就是这种）时，
    需要先把 batch 拆成单条样本做变换，再重新拼回 batch——因为 openpi 的变换
    都是按“单条样本”设计的（见 transforms.py 的接口说明）。
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                # 从任意一个叶子数组的 shape[0] 推断 batch 大小（假设所有叶子共享相同的 batch 维）。
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                # 用 tree_map 对“每棵树”做切片，保证每个元素保留原来的嵌套结构；
                # 这里的 lambda 在列表推导式内立即求值，因此闭包变量 i 不存在延迟绑定问题。
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                # 把“样本列表”重新组合成一棵树：*transformed 展开后，
                # 同一路径上的所有叶子沿新的 axis=0 堆叠回一个 batch。
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                # 本来就是逐条样本的流，直接变换即可。
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    """完全不需要真实数据的“假数据集”：按模型输入规格生成随机数据。

    用途：debug 配置、单元测试、CI 冒烟测试。它保证形状 / dtype 与真实数据一致，
    因此可以直接跑通“数据加载 → 变换 → 训练一步”的完整链路。
    """

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        # inputs_spec() 返回 (Observation 规格, Actions 规格)，两者都是 jax.ShapeDtypeStruct
        # （只有形状 / dtype、没有真实数值），并且带一个 batch 维度。
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        # 用样本下标作为随机种子：同一个 index 永远生成同一份数据，
        # 这样多 worker、多次运行的结果都可复现（方便排查数据 pipeline 的问题）。
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            # 规格里带 batch 维（batch=1），而数据集应该返回“单条样本”，因此去掉第 0 维。
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                # 浮点叶子：[-1, 1] 均匀分布，与图像 / 状态的数值范围大致一致。
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                # 整数叶子（token id 之类）：随机填 [0, 2048) 的整数，仅用于跑通形状。
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            # 其它 dtype（如 bool mask）：填 0 / False 即可，形状正确最重要。
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        # 按规格逐叶子生成假观测与假动作。
        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        # 展开成 openpi 的“嵌套 dict”数据格式：Observation 字段 + actions。
        # to_dict() 会把 images / image_masks 的键名转成 image / image_mask。
        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training.

    创建“随机访问”数据集（目前是 LeRobot 数据集或假数据）。

    Args:
        data_config: 数据配置，repo_id 指定数据集（例如 HF 上的 LeRobot 数据集）。
        action_horizon: 每次训练要预测的未来动作步数 H。
        model_config: 模型配置，仅 FakeDataset 需要（用它推断输入形状）。
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        # 没有配置数据集来源就无法创建；FakeDataConfig 用的是 "fake"。
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        # debug / 单测：不读磁盘，直接造 1024 条形状正确的假数据。
        return FakeDataset(model_config, num_samples=1024)

    # 懒加载 lerobot（可选依赖，见文件开头的说明）。
    lerobot_dataset = _load_lerobot_dataset_module()
    # 元信息里含 fps（帧率）与 tasks（任务文本映射）。
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # LeRobotDataset 本身按帧返回数据；delta_timestamps 让它在每个采样点额外返回
    # “未来 action_horizon 步”的动作序列，也就是训练目标 action chunk：
    # 第 t 步取未来 [t/fps, (t+1)/fps, ...] 这些时刻的动作。
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        # LeRobot 样本里只有 task_index（整数），这里用数据集的任务表把它翻译成 prompt 文本。
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    # RLDS 格式目前只支持 DROID（全量数据太大，LeRobot 的加载器撑不住）。
    # 注意：DroidRldsDataset 内部用 TensorFlow 流水线，且已经自己完成 batch 拼装，
    # 所以这里要把 batch_size 传进去（与后面 TorchDataLoader 再做 batching 不同）。
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    # 默认不做归一化（空 dict）；只有“真实数据集且未显式跳过”时才需要 norm_stats。
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            # 归一化统计量应当在训练前由 scripts/compute_norm_stats.py 预计算好，
            # 否则模型看到的输入 / 输出量纲与训练时不一致，训练会很不稳定。
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # 变换顺序与推理侧（policies/policy_config.py）保持一致：
    # repack（字段改名）→ data_transforms（机器人特有）→ Normalize → model_transforms。
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    # 与 transform_dataset 逻辑相同，只是面向“流式数据集”（RLDS）。
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            # 变换顺序与 transform_dataset 完全一致，保证流式 / 随机访问两条路结果相同。
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,  # True 表示底层已经给出 batch，需要先拆成单样本再变换
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    创建训练数据加载器——本文件的对外总入口，scripts/train.py 调用的就是它。

    它会根据 DataConfig 自动选择实现：
    - 配置了 `rlds_data_dir`（DROID / RLDS）→ create_rlds_data_loader；
    - 其它情况（LeRobot 数据集或 fake）→ create_torch_data_loader。

    Args:
        config: The training configuration.
            训练配置：提供 data 工厂、batch_size、seed、num_workers、
            assets_dirs 以及模型配置等。
        sharding: The sharding to use for the data loader (JAX only).
            JAX 下希望数据如何分片（通常传训练用的 data_sharding）；PyTorch 下忽略。
        shuffle: Whether to shuffle the data.
            是否打乱样本顺序（训练时通常为 True）。
        num_batches: Determines the number of batches to return.
            限制最多返回多少个 batch；None 表示无限迭代（数据集循环使用）。
        skip_norm_stats: Whether to skip data normalization.
            跳过敏捷化所需统计量的检查（测试 / 统计脚本常用）。
        framework: The framework to use ("jax" or "pytorch").
            训练框架：决定返回 JAX 分片数组还是 torch.Tensor。
    """
    # 由“数据工厂”生成真正的 DataConfig：repo_id、变换、norm_stats 等都在这一步确定
    # （有些字段依赖模型类型，必须运行时才能展开）。
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        # RLDS 分支：数据由 TensorFlow 流水线读取，且自己完成 batch 拼装。
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    # Torch / LeRobot 分支：随机访问数据集 + PyTorch DataLoader 负责 batching。
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    创建基于 PyTorch DataLoader 的加载器（LeRobot 数据集或 fake 数据）。

    本函数主要处理三件事：
    1. 组装数据集与变换（create_torch_dataset + transform_dataset）；
    2. 计算“本进程 / 本 rank 的本地 batch_size”，并处理多卡采样（DDP DistributedSampler）；
    3. 把 batch 转成 JAX 分片数组（JAX）或 torch.Tensor（PyTorch）。

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    # 1) 建立原始数据集，并套上完整的输入变换链（含归一化）。
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    # 2) 计算本地批大小：
    #    - PyTorch DDP：每个 rank 只处理 1/world_size 的数据，用 DistributedSampler 切分；
    #    - JAX 多进程：batch_size 按进程数均分；
    #    - 单卡单进程：本地批大小就是全局 batch_size。
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            # DDP 下由 DistributedSampler 负责“每个 rank 取哪部分数据”，
            # drop_last=True 保证各 rank 的样本数一致，避免集合通信时卡住。
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    # 3) 交给 TorchDataLoader：它负责多 worker、shuffle、无限迭代与设备分片。
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        # PyTorch 侧不需要 JAX sharding；JAX 侧使用调用方传入的 data_sharding。
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    创建 RLDS（目前仅 DROID）数据加载器。

    与 Torch 版本的关键区别：`DroidRldsDataset` 内部用 TensorFlow / dlimp 流水线，
    并且在数据侧就完成了 batch 拼装，因此这里不需要再 collate；
    但这也意味着必须先把 batch 拆成单条样本做 openpi 变换（见 is_batched=True）。

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        # RLDS 加载器依赖 TensorFlow，目前只支持 JAX 训练。
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    # 注意：batch_size 传给了数据集本身（TF 流水线内部 batch）。
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    # is_batched=True：底层已是 batch，先拆成单样本变换、再拼回 batch。
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation.

    对本项目来说，这个类承担“随机访问数据集 → 批量数组”的全部工作：
    - 用 torch.utils.data.DataLoader 负责 batching、shuffle、多 worker 预取；
    - 用 _collate_fn 把每条样本堆成 numpy batch；
    - 提供 __iter__，支持“数据集循环使用”和限制 batch 数量；
    - JAX 训练时把 batch 转成按设备分片的 jax.Array。
    """

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        注意：这里的“PyTorch”指的是用 PyTorch 的 DataLoader 做数据加载，
        与训练框架无关——JAX 与 PyTorch 两种训练都会用它（见 create_torch_data_loader）。

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            framework: "jax" 时输出 jax.Array（带分片），"pytorch" 时输出 torch.Tensor。
        """
        if jax.process_count() > 1:
            # 多进程 JAX 的每个进程都会各自建一个数据加载器，
            # 目前实现只保证单进程场景正确，因此这里直接报错。
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            # drop_last=True（见下）要求一个完整 batch 的样本都来自本数据集；
            # 数据集太小时会一个 batch 都取不到，这里提前给出清晰错误。
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            # 默认分片：把所有本地设备排成一根 "B"（batch）轴，batch 维依次切开。
            # 注意这是“默认兜底”；正常训练时由 create_data_loader 传入主训练 mesh 的 data_sharding。
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            # 用 spawn 而不是默认的 fork：JAX / CUDA 在 fork 出来的子进程里初始化容易出问题，
            # spawn 会启动全新的 Python 解释器，更安全（代价是启动稍慢）。
            mp_context = multiprocessing.get_context("spawn")

        # 固定随机数生成器种子，保证 shuffle 的顺序可复现。
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            # 不限制 batch 数量（无限迭代）时让 worker 常驻，避免每轮重新创建进程；
            # 有上限时（测试等）则随用随关，避免进程残留。
            persistent_workers=num_workers > 0 and num_batches is None,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            # 丢掉最后不足一个 batch 的数据：保证每个 batch 形状固定，
            # 否则最后一批的形状不同会触发 JIT 重新编译。
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0  # 已产出的 batch 数量（注意：不是样本数量）
        while True:
            # 每次数据集迭代完就重新创建迭代器，从而实现“无限循环使用数据集”。
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    # 把本进程的 numpy batch 转成 JAX 全局分片数组：
                    # 每个设备持有属于自己的那一片，后续训练步骤不需要再搬运数据。
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    # torch DataLoader 取到一个 list（长度 = local_batch_size），这里把它变成 batch 数组：
    # 先统一转成 numpy（样本里可能混有 JAX 数组），再沿新的 axis=0 堆叠。
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory.

    数据加载 worker 进程不应该占用 GPU：默认 JAX 会预分配约 75% 显存，
    在 worker 里做这件事会挤占训练进程的显存。
    """
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    # 因此这里通过环境变量让 XLA：
    #   - 不预分配显存（PREALLOCATE=false）；
    #   - 按需申请 / 释放（ALLOCATOR=platform）。
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.

    对 DROID 数据加载器的薄包装，让它和 TorchDataLoader 有相同的迭代语义：
    - 批处理已经在 DROID 数据集内部完成，这里不再 collate；
    - 数据集会无限重复，这里额外支持 num_batches 上限；
    - 把 numpy batch 转成 JAX 分片数组。
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            # 与 TorchDataLoader 相同的限制：暂不支持多进程 JAX 数据加载。
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            # 默认把所有本地设备排成一根 "B"（batch）轴做数据并行分片。
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0  # 已产出的 batch 数量
        while True:
            # 与 TorchDataLoader 相同的“循环使用数据集”逻辑：
            # 数据集迭代完就重新开始，除非已经达到 num_batches 上限。
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # 转成 JAX 分片数组（每个设备持有自己的一片）。
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    """DataLoader 的统一实现：把具体加载器与 DataConfig 绑定，并产出结构化 batch。

    本类是训练代码真正看到的那一层：
    - 对外暴露 data_config()，保存 checkpoint 时要用它拿归一化统计量；
    - 迭代时把原始 dict batch 转成 `(Observation, Actions)` 元组。
    """

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            # Observation.from_dict 负责：
            # 1. 校验 tokenized_prompt 与其 mask 成对出现；
            # 2. 把 uint8 图像从 [0, 255] 归一化到 [-1, 1]（PyTorch 模型还会顺带转成 NCHW）；
            # 3. 把普通 dict 映射成带类型 / 形状约定的 Observation 对象。
            # actions 则是数据变换处理后的动作目标（通常是归一化后的 action chunk），
            # 形状 [B, action_horizon, action_dim]。
            yield _model.Observation.from_dict(batch), batch["actions"]
