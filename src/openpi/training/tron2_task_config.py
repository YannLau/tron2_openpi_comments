"""Load TRON2 training tasks from external YAML files.

=============================================================================
背景 / Background
=============================================================================
本模块是 TRON2 训练流程的"入口配置层"。它负责：

  1. 从 YAML 文件中加载任务配置 → Tron2TaskConfig（强类型 dataclass）
  2. 验证 YAML 中没有未知字段（防止拼写错误）
  3. 将 Tron2TaskConfig 转换为 OpenPI 训练框架所需的完整 TrainConfig

工作流程示意：

  用户编写         本模块解析          本模块构建             训练框架消费
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ task.yaml │ → │ load_task()  │ → │create_train_ │ → │  训练循环    │
  │ (简单配置) │    │ → Tron2Task  │    │ config()     │    │  加载数据、  │
  │           │    │    Config    │    │ → TrainConfig│    │  创建模型、  │
  │           │    │  (已验证)    │    │  (完整配置)   │    │  开始训练    │
  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘

为什么需要 Tron2TaskConfig？
  TrainConfig 是 OpenPI 框架内部的完整配置对象，字段非常多且复杂。
  如果让用户直接编写 TrainConfig，门槛太高且容易出错。
  Tron2TaskConfig 只暴露必要的、TRON2 特有的字段，其余使用合理默认值，
  大大降低了训练任务配置的复杂度。

为什么用 YAML 而不用 Python 文件？
  - YAML 更易读、更易写，非程序员也能编辑
  - YAML 天然支持嵌套结构和注释
  - 避免配置文件中有执行任意 Python 代码的安全风险
=============================================================================
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

# PyYAML：解析 YAML 配置文件（安全模式）
import yaml

# OpenPI 内部模块：
#   pi0_config           — π₀ 模型架构配置（Pi0Config 数据类）
#   config               — 训练框架配置（TrainConfig、DataConfig 等）
#   weight_loaders       — 预训练权重的加载策略
#   transforms           — 数据变换管道（如何从原始数据中提取和重组字段）
import openpi.models.pi0_config as pi0_config
from openpi.training import config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


# ============================================================================
# Tron2TaskConfig — TRON2 训练任务配置数据类
# ============================================================================

@dataclasses.dataclass(frozen=True)
class Tron2TaskConfig:
    """TRON2 训练任务的配置数据类。

    每个字段都对应 YAML 配置文件中的一个键。这个类的实例是 "不可变的"
    (frozen=True)，创建后不能修改——确保配置在训练过程中保持一致。

    用户只需要在 YAML 中填写 name、repo_id、prompt 三个必填字段，
    其余字段都有合理的默认值。

    ── 必填字段 ──
    """

    # name：任务名称，用于标识这个训练任务
    # 会出现在实验名称、日志、checkpoint 目录中
    name: str

    # repo_id：训练数据集的标识符
    # 格式取决于数据后端，通常是一个 HuggingFace 数据集 ID 或本地路径
    # 例如："my_org/tron2_pick_and_place" 或 "./data/my_dataset"
    repo_id: str

    # prompt：任务指令文本，会在训练时作为条件输入传给模型
    # 例如："pick up the red block and place it on the table"
    # 模型会学习在给定这个 prompt 的情况下应该产生什么动作

    # ── 可选的训练参数 ──

    # weight_loader：预训练模型权重的来源路径
    # 默认值指向 π₀.5 基础模型在 Google Cloud Storage 上的 checkpoint
    # 支持多种格式：GCS 路径（gs://...）、本地路径、HuggingFace ID
    # 如果不从预训练开始，可以设为空字符串 ""
    prompt: str | None = None
    weight_loader: str = "gs://openpi-assets/checkpoints/pi05_base/params"

    # num_train_steps：总训练步数
    # 每一步 = 一个 batch 的训练（前向 + 反向传播）
    # 20,000 步是一个合理的默认值，通常需要几个小时到一天
    num_train_steps: int = 20_000

    # save_interval：每隔多少步保存一次 checkpoint
    # 保存太频繁 → 占用很多磁盘空间
    # 保存太稀疏 → 出现问题可能丢失很多训练进度
    # 1,000 步是一个折中的默认值
    save_interval: int = 1000

    # batch_size：每个训练步骤处理的样本数量
    # 较大的 batch 训练更稳定但需要更多 GPU 显存
    # 32 是考虑到常见 GPU（如 A100 40GB）的默认设置
    batch_size: int = 32
    
    fsdp_devices: int = 1
    
    # action_horizon：动作预测的"时间视野"——模型一次预测未来多少步的动作
    # 50 表示模型预测未来 50 个时间步的动作序列
    # 更大的值让模型规划更长远，但训练和推理也更昂贵

    action_horizon: int = 50

    # state_dim：机器人状态向量的维度
    # 16 = 14 个关节角度 + 2 个夹爪位置（或类似组合）
    # 这个值必须与实际数据中的状态维度匹配
    state_dim: int = 16

    # assets_base_dir：训练资源文件（如归一化统计量）的存放目录
    assets_base_dir: str = "./assets"

    # checkpoint_base_dir：训练 checkpoint（模型权重保存）的存放目录
    checkpoint_base_dir: str = "./checkpoints"

    # ── 可选的观测数据键名（data key mapping）──
    # 这些字段告诉框架从数据字典的哪个路径读取对应的数据。
    # 数据字典结构通常是嵌套的：data["observation"]["images"]["cam_high"]
    # 用点号分隔的路径表示：  "observation.images.cam_high"

    # cam_high_key：高位相机的图像数据路径
    cam_high_key: str = "observation.images.cam_high"
    # cam_left_wrist_key：左手腕相机的图像数据路径
    cam_left_wrist_key: str = "observation.images.cam_left_wrist"
    # cam_right_wrist_key：右手腕相机的图像数据路径
    cam_right_wrist_key: str = "observation.images.cam_right_wrist"
    # state_key：机器人状态数据的路径（关节角度等）
    state_key: str = "observation.state"
    # action_key：动作数据的路径（训练目标/标签）
    action_key: str = "action"

    # ── 可选的训练行为开关 ──

    # prompt_from_task：是否从数据集的 "task" 字段自动生成 prompt
    # True → 每条数据自动带一个描述性 prompt（例如数据来自什么任务）
    # False → 所有数据使用相同的 prompt（上面 prompt 字段的值）
    prompt_from_task: bool = False

    # adapt_to_pi：是否将动作格式适配为 π 模型的格式
    # 某些数据集的动作表示与 π 模型可能略有不同（如绝对位置 vs 相对偏移）
    # True → 训练时自动转换
    adapt_to_pi: bool = False

    # use_delta_joint_actions：是否使用增量关节动作（而非绝对关节角）
    # True → 动作表示"当前关节角应该变化多少"
    # False → 动作表示"目标绝对关节角是多少"
    # 增量模式通常更容易学习（输出范围小、围绕 0 分布）
    use_delta_joint_actions: bool = False

    # rtc_training_simulated_delay：RTC（Real-Time Control）模拟延迟（步数）
    # 在训练时模仿现实部署中的控制延迟，让模型学会鲁棒地应对延迟
    # None → 不模拟延迟
    # 正整数 → 在训练的动作预测中引入对应步数的延迟
    # 这是 sim-to-real 迁移的关键技巧之一
    rtc_training_simulated_delay: int | None = None


# ============================================================================
# YAML 文件读取 / YAML File Reading
# ============================================================================

def _read_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """读取一个 YAML 文件并以字典形式返回，同时校验顶层结构。

    内部辅助函数，被 load_task() 调用。

    与 deploy_config.py 中的 load_deploy_config() 功能类似，
    但这里是专门为训练任务配置文件设计的简化版本：
      - 不支持 None 路径（训练配置必须有文件）
      - 使用 expanduser() 展开 ~（但与 resolve_config_path 不同，
        不做 cwd → REPO_ROOT 的路径搜索）

    安全性：
      yaml.safe_load() 只解析基础 YAML 类型，不会执行任意代码。

    参数:
        path: YAML 文件的路径（支持 ~ 展开）

    返回:
        dict[str, Any]: 解析后的字典

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 顶层结构不是字典（mapping）

    示例:
        >>> data = _read_yaml("~/my_configs/task.yaml")
        >>> type(data)
        <class 'dict'>
    """
    # expanduser()：展开 ~ 为用户主目录路径
    # 例如 "~/task.yaml" → "/home/user/task.yaml"
    with pathlib.Path(path).expanduser().open() as f:
        # safe_load：安全的 YAML 解析
        # or {}：如果 YAML 文件为空（返回 None），用空字典替代
        data = yaml.safe_load(f) or {}

    # 校验顶层结构必须是字典
    # 防止用户手误，例如在 YAML 文件顶层写了列表
    if not isinstance(data, dict):
        raise ValueError(f"Task config must be a mapping: {path}")
    return data


# ============================================================================
# 任务配置加载 / Task Configuration Loading
# ============================================================================

def load_task(path: str | pathlib.Path) -> Tron2TaskConfig:
    """从 YAML 文件加载 TRON2 任务配置。

    这个函数做了两件关键的事情：

      1. 读取 YAML 并解析为字典
      2. 验证没有一个未知字段（防止用户拼错字段名）

    未知字段检查：
      Tron2TaskConfig 用 @dataclass 定义，它的字段名集合是已知的。
      如果 YAML 中有不属于这个集合的键，说明用户可能拼错了字段名。
      此时直接报错，列出未知字段，帮助用户快速定位问题。

      例如用户写了 batch_sizee（多了一个 e），就会收到：
        ValueError: Unknown TRON2 task fields: batch_sizee

    参数:
        path: YAML 配置文件的路径

    返回:
        Tron2TaskConfig: 验证通过的不可变配置对象

    异常:
        ValueError: 配置中有未识别的字段

    示例 YAML 文件内容：
        name: pick_and_place
        repo_id: my_org/tron2_pick_and_place
        prompt: "pick up the red block and place it on the table"
        num_train_steps: 30000
        batch_size: 64
        use_delta_joint_actions: true
    """
    data = _read_yaml(path)

    # 获取 Tron2TaskConfig 中定义的所有合法字段名
    # dataclasses.fields() 返回 Field 对象列表，.name 提取字段名
    allowed = {field.name for field in dataclasses.fields(Tron2TaskConfig)}

    # 找出 YAML 中存在、但 dataclass 中没有的键
    unknown = sorted(set(data) - allowed)

    if unknown:
        # 报错时列出所有未知字段，方便用户快速修正
        raise ValueError(f"Unknown TRON2 task fields in {path}: {', '.join(unknown)}")

    # 用字典的键值对作为关键字参数创建 Tron2TaskConfig 实例
    # **data 解包字典 → Tron2TaskConfig(name="...", repo_id="...", ...)
    # 未在 YAML 中指定的字段会自动使用 dataclass 定义的默认值
    
    task = Tron2TaskConfig(**data)
    if task.prompt_from_task and task.prompt:
        raise ValueError("prompt must not be set when prompt_from_task is true")
    if not task.prompt_from_task and not task.prompt:
        raise ValueError("prompt is required unless prompt_from_task is true")
    return task


# ============================================================================
# 训练配置构建 / Training Configuration Construction
# ============================================================================

def create_train_config(path: str | pathlib.Path, *, exp_name: str | None = None) -> _config.TrainConfig:
    """加载任务 YAML 并构建完整的 OpenPI TrainConfig 对象。

    这是本模块的"主入口"——它将简单的 Tron2TaskConfig 转换为训练框架
    所需的复杂 TrainConfig。转换过程包括：

      1. 加载和验证 YAML → Tron2TaskConfig
      2. 构建数据变换管道（RepackTransform）
         — 告诉框架如何从数据字典中提取图像、状态、动作
      3. 组装完整的 TrainConfig：
         - model：π₀ 模型架构配置（Pi0Config）
         - data：数据加载配置（LeRobotTronDataConfig）
         - weight_loader：预训练权重加载策略
         - 训练超参数（步数、保存间隔、batch size 等）

    参数:
        path:     任务 YAML 配置文件的路径
        exp_name: 实验名称（用于日志和 checkpoint 目录命名）。
                  如果未指定，使用 task.name 作为实验名称。

    返回:
        TrainConfig: OpenPI 训练框架可直接使用的完整配置对象
    """
    # 第 1 步：加载并验证 YAML → Tron2TaskConfig
    task = load_task(path)

    # ── 第 2 步：构建数据变换管道 ──
    # RepackTransform 的作用：
    #   原始数据字典可能有任意嵌套结构，训练框架需要统一的数据格式。
    #   RepackTransform 从原始数据中"提取并重命名"字段，映射为框架
    #   期望的标准格式：
    #
    #     原始数据路径（由 task 字段指定）          → 标准化名称
    #     ─────────────────────────────────────────────────────────
    #     task.cam_high_key                          → images/cam_high
    #       例："observation.images.cam_high"
    #     task.cam_left_wrist_key                    → images/cam_left_wrist
    #     task.cam_right_wrist_key                   → images/cam_right_wrist
    #     task.state_key                             → state
    #       例："observation.state"
    #     task.action_key                            → actions
    #       例："action"
    #
    #   Group 包装：
    #     即使只有一个 RepackTransform，也用 Group 包装。
    #     Group 按顺序应用多个变换，这里虽然只有一个，
    #     但为将来可能添加更多预处理步骤留了扩展空间。

    repack_structure: dict[str, Any] = {
        "images": {
            "cam_high": task.cam_high_key,
            "cam_left_wrist": task.cam_left_wrist_key,
            "cam_right_wrist": task.cam_right_wrist_key,
        },
        "state": task.state_key,
        "actions": task.action_key,
    }
    if task.prompt_from_task:
        repack_structure["prompt"] = "prompt"

    repack_transforms = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
    
    # ── 第 3 步：组装 TrainConfig ──
    return _config.TrainConfig(
        # ─── 基本标识 ───
        # name：任务名称（来自 YAML）
        name=task.name,
        # exp_name：实验名称 → 用于 wandb/tensorboard 日志和 checkpoint 目录
        # 如果未指定，默认为 task.name
        exp_name=exp_name or task.name,

        # ─── 模型配置 ───
        # Pi0Config 定义了 π₀ 模型的架构参数
        #   pi05=True → 使用 π₀.5 架构（最新的改进版）
        #   action_horizon → 一次预测多少步动作
        #   rtc_training_simulated_delay → 模拟控制延迟（提升部署鲁棒性）
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=task.action_horizon,
            rtc_training_simulated_delay=task.rtc_training_simulated_delay,
        ),

        # ─── 数据配置 ───
        # LeRobotTronDataConfig 是专门为 TRON2 数据格式设计的数据加载器配置
        #   repo_id → 数据集位置（HuggingFace ID 或本地路径）
        #   default_prompt → 训练时注入给模型的文本指令
        #   base_config → 基础数据配置（如是否从数据集的 task 字段读取 prompt）
        #   use_delta_joint_actions → 使用增量关节角还是绝对关节角
        #   adapt_to_pi → 是否将动作格式适配为 π 模型的格式
        #   state_dim → 状态向量维度
        #   repack_transforms → 上述构建的数据变换管道
        data=_config.LeRobotTronDataConfig(
            repo_id=task.repo_id,
            default_prompt=task.prompt,
            base_config=_config.DataConfig(prompt_from_task=task.prompt_from_task),
            use_delta_joint_actions=task.use_delta_joint_actions,
            adapt_to_pi=task.adapt_to_pi,
            state_dim=task.state_dim,
            repack_transforms=repack_transforms,
        ),

        # ─── 权重加载 ───
        # CheckpointWeightLoader 从指定路径加载预训练权重
        # 这允许从 π₀.5 的基础模型开始微调（transfer learning），
        # 而不是从零开始训练
        weight_loader=weight_loaders.CheckpointWeightLoader(task.weight_loader),

        # ─── 训练超参数 ───
        num_train_steps=task.num_train_steps,
        save_interval=task.save_interval,
        batch_size=task.batch_size,
        
        fsdp_devices=task.fsdp_devices,
        
        # ─── 目录配置 ───
        # assets_base_dir：资源文件目录（如归一化统计量 mean/std）
        assets_base_dir=task.assets_base_dir,
        # checkpoint_base_dir：模型权重保存目录
        checkpoint_base_dir=task.checkpoint_base_dir,
    )
