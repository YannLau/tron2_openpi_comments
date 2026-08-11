"""Run TRON2 Real-Time Chunking (RTC) deployment.

================================================================================
概述 / Overview
================================================================================
本脚本是 TRON2 真机部署的 **RTC（Real-Time Chunking）客户端**。与同步客户端
pi_client.py 的"观察→推理→执行→观察→…"串行模式不同，RTC 采用 **生产者-消费者**
双线程异步架构，实现推理和执行并行，大幅降低延迟。

================================================================================
RTC 是什么？ / What is RTC?
================================================================================
RTC（Real-Time Chunking）是一种实时动作分块策略。核心思想：

  传统的 Action Chunking：
    策略模型一次输出 H 个分步动作（如 H=50），机器人逐个执行。
    执行完 H 步后再做下一次推理。问题是：推理期间机器人"干等"。

  RTC 的做法：
    不等上一批动作执行完，在后台线程中提前发起下一次推理。新推理
    结果根据当前的执行进度，与旧动作序列的剩余部分"合并"（merge），
    形成一个平滑过渡的新动作序列。

  形象比喻：就像视频流媒体的"缓冲"机制——播放器（consumer）不断从
  缓冲区取帧播放，下载线程（producer）不断往缓冲区补充新帧，并在
  连接处做平滑过渡，确保播放不卡顿。

================================================================================
架构 / Architecture
================================================================================

  ┌─────────────────────────────────────────────────────────────────┐
  │                        Main Thread                              │
  │  - 加载配置、初始化环境                                          │
  │  - 创建 ActionQueue（生产者-消费者之间的共享队列）                │
  │  - 启动 Producer 和 Consumer 线程                                │
  │  - 监控运行状态、处理优雅退出                                    │
  └──────────┬──────────────────────────────────┬───────────────────┘
             │                                  │
    ┌────────▼──────────┐              ┌────────▼──────────┐
    │  Producer Thread  │              │  Consumer Thread  │
    │                   │              │                   │
    │ 1. 从机器人获取   │  共享队列    │ 1. 从队列取动作   │
    │    观测 (obs)     │──────────────▶│                   │
    │ 2. 发送给策略     │  ActionQueue │ 2. 执行 env.step()│
    │    服务器推理     │◀─────────────│                   │
    │ 3. merge 到队列   │  线程安全    │ 3. 按频率控制节奏 │
    │                   │              │                   │
    └───────────────────┘              └───────────────────┘

  关键术语：
  - H (action_horizon)    : 策略一次推理输出的动作总步数（如 50）
  - s (execution_horizon) : 执行窗口大小，即从队列中取多少步后触发新推理
  - d (delay)             : 推理延迟补偿（步数），用于对齐新旧动作的时间线
  - trigger_queue_size    : 队列剩余 ≤ H-s 时触发一次新推理

================================================================================
依赖关系 / Dependencies
================================================================================
- _external_tron2_env.py : 确保 TRON2 环境包在 Python 路径中
- deploy_config.py       : 部署配置加载、观测格式化、prompt 控制
- openpi_client          : OpenPI WebSocket 客户端（策略通信）
- tron2_env              : TRON2 机器人环境封装
- tron2_env.rtc          : RTC 专用组件（ActionQueue, LatencyTracker）
- numpy                  : 数值计算
- threading              : 双线程架构（Producer + Consumer）
"""

from __future__ import annotations

import argparse
from collections import deque          # 双端队列，用于滑动窗口存储推理延迟历史
from dataclasses import dataclass      # 轻量级数据类，用于配置对象
import logging
import math
import sys
from threading import Event
from threading import Thread
import time
import traceback
from threading import Event, Thread   # Event: 线程间信号; Thread: 线程

import numpy as np

# ---------------------------------------------------------------------------
# OpenPI WebSocket 客户端 —— 和远程策略服务器通信
# ---------------------------------------------------------------------------
from openpi_client import websocket_client_policy

# ---------------------------------------------------------------------------
# 本地辅助模块（examples/tron2/ 目录下）
# ---------------------------------------------------------------------------
from _external_tron2_env import ensure_external_tron2_env_on_path
from deploy_config import age_ms                  # 计算某时间戳距今的时长（毫秒）
from deploy_config import build_env_config        # 从 YAML 构建环境配置
from deploy_config import bool_value              # 安全解析布尔型配置
from deploy_config import format_obs              # 将观测格式化为策略模型的输入
from deploy_config import load_deploy_config      # 加载 YAML 配置
from deploy_config import policy_host             # 提取策略服务器主机
from deploy_config import policy_port             # 提取策略服务器端口
from deploy_config import PromptController        # 运行时动态修改 prompt
from deploy_config import record_paths            # 生成录制文件路径
from deploy_config import relative_sensor_time_s  # 计算传感器时间戳的相对时间
from deploy_config import section                 # 提取配置子段落
from deploy_config import timestamp_ms            # 安全解析时间戳（毫秒）

# ---------------------------------------------------------------------------
# 在导入 TRON2 环境前，确保外部 TRON2 包的路径已加入 sys.path
# ---------------------------------------------------------------------------
ensure_external_tron2_env_on_path()

from tron2_env import Tron2Env
from tron2_env.rtc import ActionQueue
from tron2_env.rtc import LatencyTracker

# 模块级 logger，使用 __name__ 作为标识，便于在日志中定位来源
logger = logging.getLogger(__name__)

# =============================================================================
# 全局常量 / Global Constants
# =============================================================================

# 策略模型的默认动作窗口长度（一次推理输出多少步动作）
# 典型值 50，意味着模型一次预测未来 50 步的动作轨迹
DEFAULT_ACTION_HORIZON = 50

# 原始动作向量的维度（双臂各 8 维 × 2 = 16 维？这里 32 维可能含其他信息）
ACTION_DIM_RAW = 32

# 推理延迟历史的滑动窗口大小（用于计算 P95 延迟）
INFERENCE_DELAY_HISTORY_SIZE = 10

# Producer 线程轮询队列状态的间隔（秒）
# 当队列太满（超过触发阈值）时，Producer 不会立即推理，而是等待
TRIGGER_POLL_INTERVAL_S = 0.005

# ---------------------------------------------------------------------------
# 动作索引常量：用于后续处理（平滑、EMA）时区分机械臂关节和夹爪
# ---------------------------------------------------------------------------
# 双臂的关节索引：左臂 0-6（7 个关节）+ 右臂 8-14（7 个关节）= 14 维
# 跳过索引 7（左夹爪）和索引 15（右夹爪）
PROCESSED_ARM_INDICES = tuple(range(7)) + tuple(range(8, 15))

# 夹爪索引：左夹爪=7，右夹爪=15
PROCESSED_GRIPPER_INDICES = (7, 15)


# =============================================================================
# 命令行参数解析
# =============================================================================
def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含 deploy_config 和 prompt 两个属性。
    """
    parser = argparse.ArgumentParser(description="Run TRON2 RTC deployment.")
    parser.add_argument("--profile", type=str, default=None, help="Path to client deployment profile YAML.")
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Deprecated alias for --profile.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt sent with each observation. Overrides client.prompt in YAML.",
    )
    return parser.parse_args()


# =============================================================================
# 数值安全解析辅助函数
# =============================================================================
def _safe_int(value, default: int = -1) -> int:
    """安全地将一个值转换为整数，失败或为 NaN 时返回默认值。

    在时间戳处理中，缺失或无效的值很常见（传感器可能未就绪），
    这个函数确保不会因为一个无效值而崩溃。

    Args:
        value:   待转换的值（可以是数字、字符串或 None）。
        default: 转换失败时的默认返回值。

    Returns:
        转换后的整数，或 default。
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):
        return default
    return int(value)


def _p95_int(values) -> int:
    """计算一组值的 P95（第 95 百分位数），结果转为整数。

    P95 用于估算推理延迟的"最坏情况但排除极端异常值"的水平。
    例如，10 次推理耗时 [100, 102, 105, 108, 110, 115, 120, 130, 200, 5000] ms，
    P95 不会取 5000（那可能是偶发抖动），而是一个更稳定的上界。

    Args:
        values: 数值的可迭代对象。

    Returns:
        P95 值（整数），空集合返回 0。
    """
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    # 取第 ceil(0.95 * N) - 1 个元素（0-indexed）
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


# =============================================================================
# RTC 时序参数解析
# =============================================================================
def _resolve_rtc_timing(
    client_profile: dict,
    action_horizon: int,
) -> tuple[int, int, int]:
    """从客户端配置中解析 RTC 的三个核心时序参数。

    Args:
        client_profile: client 段落的配置字典。
        action_horizon: 策略模型的动作窗口长度 H。

    Returns:
        (execution_horizon, delay, trigger_queue_size) 三元组：
        - execution_horizon (s): 从队列取多少步动作后触发新推理。
        - delay (d):              推理延迟补偿步数。
        - trigger_queue_size:     队列阈值 = H - s。

    Raises:
        ValueError: 如果执行窗口或延迟不满足约束条件。

    RTC 时序约束：
        - 0 <= s <= H  ：执行窗口不能超过总窗口
        - 0 <= d < H   ：延迟补偿步数不能超过总窗口

    直觉理解：
        H=50, s=10 时，机器人每执行 10 步动作就会触发一次新推理（此时队列
        还剩 40 步）。新推理结果会与这 40 步合并。

        d=5 意味着我们预计推理需要 5 个时间步的延迟，新动作会从第 5 步开始
        覆盖旧动作，确保新旧轨迹平滑过渡。
    """
    # execution_horizon 和 delay 是 RTC 模式必需的参数
    if "execution_horizon" not in client_profile:
        raise ValueError("client.execution_horizon is required for RTC deployment.")
    if "delay" not in client_profile:
        raise ValueError("client.delay is required for RTC deployment.")

    execution_horizon = int(client_profile["execution_horizon"])
    delay = int(client_profile["delay"])

    # 验证约束
    if not 0 <= execution_horizon <= action_horizon:
        raise ValueError(
            f"client.execution_horizon must satisfy 0 <= s <= H; "
            f"got s={execution_horizon}, H={action_horizon}."
        )
    if not 0 <= delay < action_horizon:
        raise ValueError(
            f"client.delay must satisfy 0 <= d < H; "
            f"got d={delay}, H={action_horizon}."
        )

    # 触发阈值：队列中最多保留 H-s 步（一旦低于此值就触发新推理）
    trigger_queue_size = action_horizon - execution_horizon
    return execution_horizon, delay, trigger_queue_size


def _resolve_guidance_weight(client_profile: dict) -> float:
    """解析 RTC guidance 权重。

    RTC guidance 是一种对新旧动作序列进行加权插值的机制。
    weight 越大，新推理结果对最终动作的影响越大。

    Args:
        client_profile: client 段落的配置字典。

    Returns:
        rtc_guidance_weight（正浮点数）。

    Raises:
        ValueError: 如果权重 <= 0。
    """
    weight = float(client_profile.get("rtc_guidance_weight", 10.0))
    if weight <= 0:
        raise ValueError(f"client.rtc_guidance_weight must be positive, got {weight}")
    return weight


# =============================================================================
# ActionPostProcessor —— 客户端动作后处理（平滑）
# =============================================================================

@dataclass(frozen=True)
class ActionPostprocessConfig:
    """动作后处理的配置（不可变数据类）。

    Attributes:
        enabled:               是否启用后处理。
        boundary_blend_frames: 新旧动作序列边界处的混合帧数。
        boundary_blend_curve:  混合曲线类型 —— "linear"（线性）或 "smoothstep"（平滑阶梯）。
        boundary_blend_scope:  混合范围 —— "arm"（仅关节）、"gripper"（仅夹爪）、"all"（全部）。
        ema_alpha:             EMA（指数移动平均）的平滑系数，1.0 表示不平滑。
        ema_frames:            EMA 应用的帧数，0 表示应用到所有帧。
        ema_scope:             EMA 范围 —— "arm"、"gripper"、"all"。
    """
    enabled: bool = False
    boundary_blend_frames: int = 0
    boundary_blend_curve: str = "smoothstep"
    boundary_blend_scope: str = "arm"
    ema_alpha: float = 1.0
    ema_frames: int = 0
    ema_scope: str = "arm"


class ActionPostProcessor:
    """可选的客户端动作平滑处理器。

    在 RTC 模式下，新旧动作序列在边界处可能存在跳变（新旧预测不一致）。
    ActionPostProcessor 提供两种平滑机制：

    1. Boundary Blend（边界混合）：
       在新旧动作序列的交界处，用加权插值平滑过渡。类似于
       两个视频片段之间的"淡入淡出"转场效果。

       blend_frames=5 时，前 5 帧是旧动作到新动作的渐变：
         frame[0] = 0.83*old + 0.17*new
         frame[1] = 0.67*old + 0.33*new
         frame[2] = 0.50*old + 0.50*new
         frame[3] = 0.33*old + 0.67*new
         frame[4] = 0.17*old + 0.83*new

       当 curve="smoothstep" 时，alpha 会经过 smoothstep 函数处理：
         alpha' = alpha² * (3 - 2*alpha)
       这使得过渡在开始和结束时更平缓（导数连续）。

    2. EMA（指数移动平均）：
       对动作序列逐帧做指数平滑，减少高频抖动：
         filtered[t] = alpha * current[t] + (1-alpha) * filtered[t-1]

       alpha=0.8 时，新值占 80%，历史值占 20%
       alpha=0.3 时，新值占 30%，历史值占 70%（更平滑但响应慢）

    两种机制可以叠加使用：先 Boundary Blend，再 EMA。
    """

    def __init__(self, config: ActionPostprocessConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        """后处理是否真正生效。

        只有当 enabled=True 并且至少有一个有效的平滑参数时才返回 True。
        """
        return self.config.enabled and (
            self.config.boundary_blend_frames > 0 or self.config.ema_alpha < 1.0
        )

    def describe(self) -> str:
        """返回后处理配置的可读描述字符串，用于日志输出。

        Example: "blend=5:arm:smoothstep+ema=0.80:arm:all"
        """
        if not self.enabled:
            return "off"
        parts = []
        if self.config.boundary_blend_frames > 0:
            parts.append(
                f"blend={self.config.boundary_blend_frames}:"
                f"{self.config.boundary_blend_scope}:"
                f"{self.config.boundary_blend_curve}"
            )
        if self.config.ema_alpha < 1.0:
            parts.append(
                f"ema={self.config.ema_alpha:.2f}:{self.config.ema_scope}:"
                f"{self.config.ema_frames or 'all'}"
            )
        return "+".join(parts)

    def apply(
        self,
        processed_actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        merge_delay: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """对处理后的动作序列应用平滑。

        Args:
            processed_actions:      当前推理输出的动作序列，形状 (N, action_dim)。
            old_processed_leftover: 上一批次剩余的动作序列（用于边界混合的旧值）。
            merge_delay:            新旧动作交界处在新序列中的起始索引。

        Returns:
            (smoothed_actions, diagnostics) 元组：
            - smoothed_actions: 平滑后的动作序列（与输入同形状）。
            - diagnostics:      诊断信息字典（MAE、最大差异、应用帧数等）。
        """
        # 初始化诊断信息
        diagnostics = {
            "action_postprocess_enabled": float(self.enabled),
            "action_postprocess_merge_delay": float(max(0, merge_delay)),
            "action_postprocess_blend_frames": 0.0,
            "action_postprocess_ema_frames": 0.0,
            "action_postprocess_mae": 0.0,
            "action_postprocess_max": 0.0,
        }
        if not self.enabled:
            return processed_actions, diagnostics

        # 保留原始副本用于计算差异诊断
        original = np.asarray(processed_actions)
        actions = original.copy()

        # merge_delay 表示新旧动作序列的交界位置
        # 例如 merge_delay=3 表示前 3 帧是旧动作的延续，从第 3 帧开始是新动作
        start = max(0, min(int(merge_delay), len(actions)))

        # 按顺序应用两种平滑
        blend_frames = self._apply_boundary_blend(actions, old_processed_leftover, start)
        ema_frames = self._apply_ema(actions, old_processed_leftover, start)

        # 计算平滑前后动作的差异，用于监控平滑幅度是否合理
        delta = np.abs(actions.astype(np.float64) - original.astype(np.float64))
        diagnostics.update(
            {
                "action_postprocess_blend_frames": float(blend_frames),
                "action_postprocess_ema_frames": float(ema_frames),
                "action_postprocess_mae": float(np.mean(delta)) if delta.size else 0.0,
                "action_postprocess_max": float(np.max(delta)) if delta.size else 0.0,
            }
        )
        return actions, diagnostics

    def _apply_boundary_blend(
        self,
        actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        start: int,
    ) -> int:
        """在新旧动作序列交界处做加权混合，平滑过渡。

        Args:
            actions:                当前动作序列（会原地修改）。
            old_processed_leftover: 上一批剩余动作。
            start:                  混合起始索引。

        Returns:
            实际混合的帧数。
        """
        frames = self.config.boundary_blend_frames
        if frames <= 0 or old_processed_leftover is None or start >= len(actions):
            return 0

        # 混合帧数不能超过新旧动作序列的长度
        count = min(frames, len(actions) - start, len(old_processed_leftover))
        if count <= 0:
            return 0

        # 确定哪些维度需要混合（由 scope 决定）
        action_dim = min(actions.shape[1], old_processed_leftover.shape[1])
        dims = _processed_scope_indices(self.config.boundary_blend_scope, action_dim)
        if dims.size == 0:
            return 0

        # 逐帧混合
        for offset in range(count):
            # alpha 从 1/(count+1) 到 count/(count+1)
            # 第 0 帧 alpha 最小（偏向旧动作），最后一帧 alpha 最大（偏向新动作）
            alpha = (offset + 1) / (count + 1)

            # smoothstep 曲线：使过渡开始和结束更平缓
            # f(t) = 3t² - 2t³，在 t=0 和 t=1 处导数为 0
            if self.config.boundary_blend_curve == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            # 线性插值：result = (1-alpha)*old + alpha*new
            actions[start + offset, dims] = (
                (1.0 - alpha) * old_processed_leftover[offset, dims]
                + alpha * actions[start + offset, dims]
            )
        return count

    def _apply_ema(
        self,
        actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        start: int,
    ) -> int:
        """对动作序列应用指数移动平均（EMA）滤波。

        EMA 可以抑制动作序列中的高频抖动，使运动更平滑。
        缺点是会引入一定延迟（alpha 越小越平滑但延迟越大）。

        Args:
            actions:                当前动作序列（会原地修改）。
            old_processed_leftover: 上一批剩余动作（提供 EMA 的初始值）。
            start:                  EMA 起始索引。

        Returns:
            实际应用 EMA 的帧数。
        """
        alpha = self.config.ema_alpha
        if alpha >= 1.0 or start >= len(actions):
            return 0

        count = len(actions) - start
        if self.config.ema_frames > 0:
            count = min(count, self.config.ema_frames)
        if count <= 0:
            return 0

        # 确定 EMA 作用的维度范围
        action_dim = actions.shape[1]
        if old_processed_leftover is not None and len(old_processed_leftover) > 0:
            action_dim = min(action_dim, old_processed_leftover.shape[1])
        dims = _processed_scope_indices(self.config.ema_scope, action_dim)
        if dims.size == 0:
            return 0

        # 初始化 EMA 的"前一个值"
        if old_processed_leftover is not None and len(old_processed_leftover) > 0:
            # 用上一批最后一个动作作为 EMA 的初始值
            previous = old_processed_leftover[0, dims].astype(np.float64, copy=True)
        else:
            # 没有历史值时，用当前序列的第一个值作为初始值
            previous = actions[start, dims].astype(np.float64, copy=True)

        # 逐帧 EMA：filtered[i] = alpha * current[i] + (1-alpha) * filtered[i-1]
        for offset in range(count):
            current = actions[start + offset, dims].astype(np.float64, copy=False)
            filtered = alpha * current + (1.0 - alpha) * previous
            actions[start + offset, dims] = filtered
            previous = filtered
        return count


# =============================================================================
# 动作后处理配置解析
# =============================================================================
def _resolve_action_postprocess(client_profile: dict) -> ActionPostProcessor:
    """从客户端配置中解析动作后处理配置。

    支持新旧两套配置键名：
    - 新键名（推荐）：client.rtc_action_postprocess 下的子字段
    - 旧键名（兼容）：client 下的扁平字段（如 rtc_action_postprocess_enabled）

    Args:
        client_profile: client 段落的配置字典。

    Returns:
        配置好的 ActionPostProcessor 实例。

    Raises:
        ValueError: 如果参数不合法。
    """
    raw_config = client_profile.get("rtc_action_postprocess", {}) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("client.rtc_action_postprocess must be a mapping when provided.")

    # 辅助函数：优先取新配置键，回退到旧键，都不存在则用默认值
    def option(name: str, legacy_name: str, default):
        return raw_config.get(name, client_profile.get(legacy_name, default))

    config = ActionPostprocessConfig(
        enabled=bool_value(option("enabled", "rtc_action_postprocess_enabled", False)),
        boundary_blend_frames=max(
            0, int(option("boundary_blend_frames", "rtc_boundary_blend_frames", 0))
        ),
        boundary_blend_curve=str(
            option("boundary_blend_curve", "rtc_boundary_blend_curve", "smoothstep")
        ),
        boundary_blend_scope=str(
            option("boundary_blend_scope", "rtc_boundary_blend_scope", "arm")
        ),
        ema_alpha=float(option("ema_alpha", "rtc_action_ema_alpha", 1.0)),
        ema_frames=max(0, int(option("ema_frames", "rtc_action_ema_frames", 0))),
        ema_scope=str(option("ema_scope", "rtc_action_ema_scope", "arm")),
    )

    # 参数合法性验证
    if config.boundary_blend_curve not in {"linear", "smoothstep"}:
        raise ValueError(
            "client.rtc_action_postprocess.boundary_blend_curve must be 'linear' or 'smoothstep'."
        )
    for field_name, scope in (
        ("boundary_blend_scope", config.boundary_blend_scope),
        ("ema_scope", config.ema_scope),
    ):
        if scope not in {"arm", "gripper", "all"}:
            raise ValueError(
                f"client.rtc_action_postprocess.{field_name} must be 'arm', 'gripper', or 'all'."
            )
    if not 0.0 < config.ema_alpha <= 1.0:
        raise ValueError(
            "client.rtc_action_postprocess.ema_alpha must satisfy 0 < alpha <= 1."
        )

    return ActionPostProcessor(config)


def _processed_scope_indices(scope: str, action_dim: int) -> np.ndarray:
    """根据 scope 返回应被后处理的动作维度索引。

    Args:
        scope:      范围 —— "all"（全部）、"arm"（仅关节）、"gripper"（仅夹爪）。
        action_dim: 动作向量的总维度。

    Returns:
        需要处理的维度索引数组（0-indexed）。
    """
    if scope == "all":
        return np.arange(action_dim, dtype=np.int64)
    if scope == "arm":
        indices = PROCESSED_ARM_INDICES     # (0-6, 8-14)
    elif scope == "gripper":
        indices = PROCESSED_GRIPPER_INDICES # (7, 15)
    else:
        return np.empty((0,), dtype=np.int64)
    # 过滤掉超出 action_dim 的索引（安全措施）
    return np.asarray([idx for idx in indices if idx < action_dim], dtype=np.int64)


# =============================================================================
# warmup_rtc —— RTC 预热推理
# =============================================================================
def warmup_rtc(
    ws_client: websocket_client_policy.WebsocketClientPolicy,
    env: Tron2Env,
    prompt_controller: PromptController,
    rtc_guidance_enabled: bool,
    rtc_guidance_weight: float,
    delay: int,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    trained_rtc_mode: bool = False,
):
    """在正式控制循环开始前，执行预热推理。

    为什么需要预热？
    1. GPU/模型首次推理通常较慢（冷启动），预热可以消除这个影响。
    2. 预热结果可以预先填充 ActionQueue，让 Consumer 一开始就有数据
       可取，避免启动时的空队列停顿。
    3. 预热过程中验证 RTC 路径正常通信。

    预热分两步：
      #1: 普通（非 RTC）推理 —— 验证基本通信畅通
      #2: RTC 推理 —— 验证 RTC 参数和 guidance 路径工作正常

    Args:
        ws_client:            WebSocket 策略客户端。
        env:                  机器人环境。
        prompt_controller:    Prompt 控制器（获取当前任务指令）。
        rtc_guidance_enabled: 是否启用 RTC guidance。
        rtc_guidance_weight:  RTC guidance 权重。
        delay:                RTC 延迟补偿步数。
        action_horizon:       动作窗口长度 H。
        trained_rtc_mode:     是否使用训练好的原生 RTC 模式（而非 guidance）。

    Returns:
        第一次非 RTC 推理的结果字典（用于后续填充队列）。
    """
    logger.info("[WARMUP] Starting RTC warmup...")

    # 获取一次观测并用当前 prompt 格式化
    obs = env.get_obs()
    obs_formatted = format_obs(obs, prompt=prompt_controller.get())

    # ---- 第一步：非 RTC 普通推理 ----
    t0 = time.perf_counter()
    base_result = ws_client.infer(obs_formatted, return_raw_actions=True)
    t1 = time.perf_counter()
    base_result["client_warmup_latency_s"] = t1 - t0
    logger.info("[WARMUP #1] Non-RTC: %.1fms", (t1 - t0) * 1000)

    # ---- 准备一个全零的 dummy leftover 用于 RTC 推理 ----
    # pretrained_rtc_mode 和 guidance 模式都需要传入 prev_chunk_left_over
    dummy_left_over = np.zeros((action_horizon, ACTION_DIM_RAW), dtype=np.float32)

    # ---- 第二步：RTC 推理（根据模式构建不同的参数） ----
    if trained_rtc_mode:
        # 训练好的原生 RTC 模式：策略模型内部已包含 RTC 逻辑
        rtc_kwargs = {
            "trained_rtc_mode": True,
            "inference_delay": delay,
            "return_raw_actions": True,
            "prev_chunk_left_over": dummy_left_over,
        }
    elif rtc_guidance_enabled:
        # RTC guidance 模式：客户端侧对推理结果做 guidance 加权融合
        rtc_kwargs = {
            "inference_delay": delay,
            "max_guidance_weight": rtc_guidance_weight,
            "prefix_horizon": 0,           # 无前缀（预热时没有历史动作）
            "return_raw_actions": True,
            "prev_chunk_left_over": dummy_left_over,
            "prev_chunk_left_over_len": 0, # 无历史剩余长度
        }
    else:
        # guidance 关闭时的 replace-only 模式：直接用非 RTC 推理结果
        logger.info(
            "[WARMUP] RTC guidance disabled; replace-only mode uses non-RTC warmup actions."
        )
        return base_result

    t0 = time.perf_counter()
    rtc_result = ws_client.infer(obs_formatted, **rtc_kwargs)
    t1 = time.perf_counter()
    logger.info(
        "[WARMUP #2] RTC path: %.1fms, raw=%s, proc=%s",
        (t1 - t0) * 1000,
        rtc_result["raw_actions"].shape if "raw_actions" in rtc_result else "N/A",
        np.stack(rtc_result["actions"], axis=0).shape,
    )
    logger.info("[WARMUP] Complete.")
    return base_result


# =============================================================================
# inference_producer —— 生产者线程
# =============================================================================
def inference_producer(
    ws_client: websocket_client_policy.WebsocketClientPolicy,
    env: Tron2Env,
    action_queue: ActionQueue,
    latency_stats: LatencyTracker,
    shutdown_event: Event,
    fps: float,
    execution_horizon: int,
    delay: int,
    rtc_guidance_enabled: bool,
    rtc_guidance_weight: float,
    trigger_queue_size: int,
    action_postprocessor: ActionPostProcessor,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    record_states: list | None = None,
    time_origin: float | None = None,
    perf_origin: float | None = None,
    trained_rtc_mode: bool = False,
    prompt_controller: PromptController | None = None,
    obs_timeout_budget_s: float = 5.0,
):
    """生产者线程：观测 → 推理 → 合并到 ActionQueue。

    这个线程运行在一个独立的 OS 线程中，和 Consumer 线程并行工作。

    工作流程：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 检查队列剩余长度 > trigger_queue_size？              │
    │    ├─ 是 → sleep 等待 Consumer 消费                      │
    │    └─ 否 → 继续                                         │
    │ 2. 从机器人环境获取观测（带超时重试）                     │
    │ 3. 格式化观测 → 发送给策略服务器推理                      │
    │ 4. 对推理结果做后处理（Boundary Blend + EMA）             │
    │ 5. action_queue.merge() 将新旧动作合并到队列              │
    │ 6. 重复                                                 │
    └─────────────────────────────────────────────────────────┘

    关键设计决策：
    - 动态延迟估计：使用最近 N 次推理的 P95 延迟作为推断延迟，
      而不是固定的 d 值。这样能自适应网络波动和 GPU 负载变化。
    - 超时保护：如果 get_obs() 持续超时，会最终报错退出，而不是
      无限等待。

    Args:
        ws_client:            WebSocket 策略客户端。
        env:                  机器人环境（用于获取观测）。
        action_queue:         RTC 动作队列（生产者-消费者共享）。
        latency_stats:        延迟统计器（跟踪推理耗时）。
        shutdown_event:       关闭信号事件。
        fps:                  控制频率（Hz）。
        execution_horizon:    执行窗口 s。
        delay:               初始推理延迟补偿 d。
        rtc_guidance_enabled: 是否启用 RTC guidance。
        rtc_guidance_weight:  RTC guidance 权重。
        trigger_queue_size:   触发推理的队列阈值。
        action_postprocessor: 动作后处理器。
        action_horizon:       动作窗口 H。
        record_states:        录制状态数据的列表（会被原地修改）。
        time_origin:          时间原点（wall clock 基准）。
        perf_origin:          性能计数器原点。
        trained_rtc_mode:     是否使用训练好的原生 RTC 模式。
        prompt_controller:    Prompt 控制器。
        obs_timeout_budget_s: 观测获取超时的总预算（秒）。
    """
    try:
        logger.info("[PRODUCER] Starting inference producer thread")

        # 每个动作分块的时长（秒）
        # 例如 fps=10 时，time_per_chunk = 0.1s
        time_per_chunk = 1.0 / fps

        infer_count = 0  # 推理计数

        # 初始延迟：由配置指定，限制在 [0, H-1] 范围内
        initial_delay = max(0, min(int(delay), action_horizon - 1))

        # 滑动窗口：存储最近 N 次推理的延迟，用于 P95 估计
        inference_delay_buffer = deque(maxlen=INFERENCE_DELAY_HISTORY_SIZE)

        # =====================================================================
        # Producer 主循环
        # =====================================================================
        while not shutdown_event.is_set():
            # ------------------------------------------------------------------
            # 步骤 1：背压控制 —— 队列太满就等待
            # 如果队列中还剩很多动作（> trigger_queue_size），说明 Consumer
            # 来不及消费，此时 Producer 应该等待而不是无脑推理。
            # ------------------------------------------------------------------
            if action_queue.qsize() > trigger_queue_size:
                # 休眠时间取触发轮询间隔和 1/4 分块时间的较大值
                time.sleep(min(TRIGGER_POLL_INTERVAL_S, time_per_chunk * 0.25))
                continue

            # ------------------------------------------------------------------
            # 步骤 2：动态估计推理延迟
            # 使用历史推理延迟的 P95 作为当前延迟估计。
            # 首次推理时使用配置的初始 d 值。
            # ------------------------------------------------------------------
            merge_delay_cap = max(0, action_horizon - 1)
            inference_delay_p95 = _p95_int(inference_delay_buffer)
            inference_delay = (
                min(inference_delay_p95, merge_delay_cap)
                if inference_delay_buffer
                else initial_delay
            )

            # ------------------------------------------------------------------
            # 步骤 3：获取观测（带超时重试）
            # 从机器人获取当前状态。如果传感器未就绪，可能会抛出
            # TimeoutError。这里用重试循环处理，但设置了总超时预算。
            # ------------------------------------------------------------------
            obs_request_perf = time.perf_counter()
            obs = None
            obs_wait_start = time.perf_counter()
            while not shutdown_event.is_set():
                try:
                    obs = env.get_obs()
                    break
                except TimeoutError as exc:
                    waited = time.perf_counter() - obs_wait_start
                    if waited >= obs_timeout_budget_s:
                        # 超过总预算 → 致命错误，退出
                        logger.error(
                            "[PRODUCER] get_obs timed out for %.1fs, exceeding %.1fs.",
                            waited,
                            obs_timeout_budget_s,
                        )
                        raise
                    # 还在预算内 → 重试
                    logger.warning(
                        "[PRODUCER] get_obs timeout %.1f/%.1fs; "
                        "waiting for a fresh observation: %s",
                        waited,
                        obs_timeout_budget_s,
                        exc,
                    )
            if obs is None:
                # 收到关闭信号或获取失败 → 退出循环
                break

            obs_receive_perf = time.perf_counter()
            obs_receive_wall = time.time()

            # 格式化观测为策略模型的输入格式
            obs_formatted = format_obs(
                obs,
                prompt=prompt_controller.get() if prompt_controller is not None else None,
            )

            # ------------------------------------------------------------------
            # 步骤 4：提取观测元数据（时间戳对齐，用于诊断和日志）
            # ------------------------------------------------------------------
            metadata = obs.get("metadata", {})
            image_timestamps = metadata.get("image_timestamps_ms", {}) or {}

            # 计算相对时间（从 origin 开始算，便于多线程时间戳对齐）
            obs_receive_time_s = obs_receive_wall - (time_origin or obs_receive_wall)
            obs_receive_perf_s = obs_receive_perf - (perf_origin or obs_receive_perf)

            # 解析各类传感器时间戳
            ref_timestamp_ms = timestamp_ms(
                metadata.get("observation_ref_timestamp_ms",
                             metadata.get("bridge_ref_timestamp_ms"))
            )
            joint_timestamp_ms = timestamp_ms(metadata.get("joint_timestamp_ms"))
            gripper_timestamp_ms = timestamp_ms(metadata.get("gripper_timestamp_ms"))
            image_timestamp_ms = timestamp_ms(metadata.get("image_timestamp_ms"))
            cam_high_timestamp_ms = timestamp_ms(image_timestamps.get("cam_high"))
            cam_left_wrist_timestamp_ms = timestamp_ms(image_timestamps.get("cam_left_wrist"))
            cam_right_wrist_timestamp_ms = timestamp_ms(image_timestamps.get("cam_right_wrist"))

            # 计算传感器数据"年龄"（从采集到被处理经过了多长时间）
            # 这是衡量系统延迟的关键指标
            obs_age_ms = age_ms(obs_receive_wall, ref_timestamp_ms)
            obs_joint_age_ms = age_ms(obs_receive_wall, joint_timestamp_ms)
            obs_gripper_age_ms = age_ms(obs_receive_wall, gripper_timestamp_ms)
            obs_image_age_ms = age_ms(obs_receive_wall, image_timestamp_ms)

            # ------------------------------------------------------------------
            # 步骤 5：获取队列快照 —— 了解当前队列中有多少剩余动作
            # ------------------------------------------------------------------
            action_index_before, prev_left_over, queue_size_before = (
                action_queue.snapshot_left_over()
            )
            prev_left_over_len = (
                int(prev_left_over.shape[0]) if prev_left_over is not None else 0
            )
            # paper_s: 论文中定义的 s 参数——需要新推理的动作步数
            # 如果队列中还有 prev_left_over_len 步旧动作，只需补充 H - prev_left_over_len 步
            paper_s = max(0, action_horizon - prev_left_over_len)
            # prefix_horizon: 旧动作前缀的长度（用于 guidance 计算）
            prefix_horizon = prev_left_over_len

            # ------------------------------------------------------------------
            # 步骤 6：录制状态数据（如果开启）
            # 保存丰富的元数据用于离线分析和调试
            # ------------------------------------------------------------------
            if record_states is not None:
                image_values = [
                    value
                    for value in (
                        cam_high_timestamp_ms,
                        cam_left_wrist_timestamp_ms,
                        cam_right_wrist_timestamp_ms,
                    )
                    if not math.isnan(value)
                ]
                record_states.append(
                    {
                        "infer_index": infer_count,
                        "queue_size": queue_size_before,
                        "action_index_before": action_index_before,
                        "initial_delay": initial_delay,
                        "inference_delay_p95": inference_delay_p95,
                        "inference_delay": inference_delay,
                        "action_horizon": action_horizon,
                        "execution_horizon": execution_horizon,
                        "rtc_guidance_enabled": float(rtc_guidance_enabled),
                        "rtc_guidance_weight": rtc_guidance_weight,
                        "action_postprocess_enabled": float(action_postprocessor.enabled),
                        "trigger_queue_size": trigger_queue_size,
                        "paper_s": paper_s,
                        "prefix_horizon": prefix_horizon,
                        "prev_left_over_len": prev_left_over_len,
                        "obs_request_perf_s": obs_request_perf
                        - (perf_origin or obs_request_perf),
                        "obs_receive_time_s": obs_receive_time_s,
                        "obs_receive_perf_s": obs_receive_perf_s,
                        "obs_wait_ms": (obs_receive_perf - obs_request_perf) * 1000.0,
                        "obs_bridge_ref_timestamp_ms": ref_timestamp_ms,
                        "obs_joint_timestamp_ms": joint_timestamp_ms,
                        "obs_gripper_timestamp_ms": gripper_timestamp_ms,
                        "obs_image_timestamp_ms": image_timestamp_ms,
                        "obs_cam_high_timestamp_ms": cam_high_timestamp_ms,
                        "obs_cam_left_wrist_timestamp_ms": cam_left_wrist_timestamp_ms,
                        "obs_cam_right_wrist_timestamp_ms": cam_right_wrist_timestamp_ms,
                        "obs_age_ms": obs_age_ms,
                        "obs_joint_age_ms": obs_joint_age_ms,
                        "obs_gripper_age_ms": obs_gripper_age_ms,
                        "obs_image_age_ms": obs_image_age_ms,
                        "obs_sensor_time_s": relative_sensor_time_s(
                            obs_receive_time_s, obs_age_ms
                        ),
                        "obs_joint_sensor_time_s": relative_sensor_time_s(
                            obs_receive_time_s, obs_joint_age_ms
                        ),
                        "obs_joint_ref_offset_ms": (
                            joint_timestamp_ms - ref_timestamp_ms
                            if not math.isnan(joint_timestamp_ms)
                            and not math.isnan(ref_timestamp_ms)
                            else float("nan")
                        ),
                        "obs_gripper_ref_offset_ms": (
                            gripper_timestamp_ms - ref_timestamp_ms
                            if not math.isnan(gripper_timestamp_ms)
                            and not math.isnan(ref_timestamp_ms)
                            else float("nan")
                        ),
                        "obs_image_span_ms": (
                            max(image_values) - min(image_values)
                            if image_values
                            else float("nan")
                        ),
                        "state": obs["state"].copy(),
                    }
                )

            # ------------------------------------------------------------------
            # 步骤 7：准备 RTC 推理参数
            # ------------------------------------------------------------------
            rtc_kwargs = {"return_raw_actions": True}

            # 训练好的原生 RTC 模式：策略模型内部处理 RTC 逻辑
            if trained_rtc_mode:
                rtc_kwargs["trained_rtc_mode"] = True
                rtc_kwargs["inference_delay"] = inference_delay

            # RTC guidance 模式：客户端侧做 guidance 加权融合
            if rtc_guidance_enabled:
                rtc_kwargs.update(
                    {
                        "inference_delay": inference_delay,
                        "max_guidance_weight": rtc_guidance_weight,
                        "prefix_horizon": prefix_horizon,
                    }
                )

            # 如果有上一批剩余动作，传给服务器做时间对齐
            if (rtc_guidance_enabled or trained_rtc_mode) and prev_left_over is not None:
                # 确保 leftover 形状为 (H, action_dim)，不足部分填零
                if prev_left_over.shape[0] < action_horizon:
                    padded = np.zeros(
                        (action_horizon, prev_left_over.shape[1]),
                        dtype=prev_left_over.dtype,
                    )
                    padded[: prev_left_over.shape[0]] = prev_left_over
                    prev_left_over = padded
                rtc_kwargs["prev_chunk_left_over"] = prev_left_over
                if rtc_guidance_enabled:
                    rtc_kwargs["prev_chunk_left_over_len"] = prev_left_over_len

            # ------------------------------------------------------------------
            # 步骤 8：执行推理
            # ------------------------------------------------------------------
            t_infer_start = time.perf_counter()
            result = ws_client.infer(obs_formatted, **rtc_kwargs)
            t_infer_end = time.perf_counter()

            # 提取原始动作和处理后的动作
            # raw_actions:   策略模型原始输出（用于 merge 和 guidance 计算）
            # actions:       经过服务器后处理的动作（用于实际执行）
            original_actions = result["raw_actions"]
            processed_actions = np.stack(result["actions"], axis=0)

            # ------------------------------------------------------------------
            # 步骤 9：更新延迟统计
            # ------------------------------------------------------------------
            new_latency = t_infer_end - t_infer_start

            # 测量推理延迟对应的时间步数
            # 例如：推理耗时 0.35s，time_per_chunk=0.1s → measured_delay=4 步
            measured_inference_delay = (
                0
                if queue_size_before <= 0
                else min(math.ceil(new_latency / time_per_chunk), merge_delay_cap)
            )
            # 首次推理的延迟数据不准确（可能有冷启动），跳过
            if infer_count > 0:
                inference_delay_buffer.append(measured_inference_delay)
            latency_stats.add(new_latency)

            # ------------------------------------------------------------------
            # 步骤 10：客户端动作后处理（Boundary Blend + EMA）
            # 在 merge 到队列之前，先对新动作做平滑处理
            # ------------------------------------------------------------------
            if action_index_before is None:
                # 首次推理（队列为空），延迟 = 测量值
                postprocess_delay = measured_inference_delay
            else:
                # 计算从上次快照到现在的索引差值
                postprocess_delay = max(
                    0, action_queue.get_action_index() - action_index_before
                )

            # 获取上一批的剩余动作，用于边界混合
            old_processed_leftover = (
                action_queue.get_processed_left_over()
                if action_postprocessor.enabled
                else None
            )
            processed_actions, postprocess_diagnostics = action_postprocessor.apply(
                processed_actions,
                old_processed_leftover,
                postprocess_delay,
            )

            # ------------------------------------------------------------------
            # 步骤 11：合并到 ActionQueue
            # merge 是 RTC 的核心操作——将新旧动作序列智能地拼接在一起
            # ------------------------------------------------------------------
            used_delay = action_queue.merge(
                original_actions,
                processed_actions,
                measured_inference_delay,
                action_index_before,
                extra_delay=0,
            )
            merge_diagnostics = action_queue.get_last_merge_diagnostics()

            # 更新录制数据
            if record_states is not None and record_states:
                record_states[-1].update(merge_diagnostics)
                record_states[-1].update(postprocess_diagnostics)
                record_states[-1].update(
                    {
                        "measured_inference_delay": measured_inference_delay,
                        "used_delay": used_delay,
                        "inference_delay_minus_used": inference_delay - used_delay,
                    }
                )

            # 延迟不足警告：实际使用的延迟超过了预估值
            if rtc_guidance_enabled and used_delay > inference_delay:
                logger.warning(
                    "RTC delay underflow: d=%d, used_delay=%d. "
                    "Delay follows recent measured p95.",
                    inference_delay,
                    used_delay,
                )

            # ------------------------------------------------------------------
            # 步骤 12：日志输出
            # ------------------------------------------------------------------
            server_timing = result.get("server_timing", {})
            policy_timing = result.get("policy_timing", {})
            logger.info(
                "[PRODUCER #%d] e2e=%.1fms server=%.1fms policy=%.1fms "
                "s=%d/H=%d guide=%s d=%d p95=%d meas=%d used=%d left=%d "
                "queue=%d->%d post=%s",
                infer_count,
                new_latency * 1000,
                server_timing.get("infer_ms", 0),
                policy_timing.get("infer_ms", 0),
                paper_s,
                action_horizon,
                rtc_guidance_enabled,
                inference_delay,
                inference_delay_p95,
                measured_inference_delay,
                used_delay,
                prev_left_over_len,
                queue_size_before,
                action_queue.qsize(),
                action_postprocessor.describe(),
            )
            infer_count += 1

        logger.info("[PRODUCER] Shutting down after %d inferences", infer_count)
    except Exception:
        # 生产者线程中的任何未捕获异常都会导致整个系统终止
        logger.error("[PRODUCER] Fatal exception:\n%s", traceback.format_exc())
        shutdown_event.set()  # 通知 Consumer 也退出
        sys.exit(1)


# =============================================================================
# control_consumer —— 消费者线程
# =============================================================================
def control_consumer(
    env: Tron2Env,
    action_queue: ActionQueue,
    shutdown_event: Event,
    fps: float,
    *,
    record_data: list | None = None,
    time_origin: float | None = None,
    perf_origin: float | None = None,
    recovery_blend_frames: int = 6,
):
    """消费者线程：从 ActionQueue 取动作 → env.step() 执行。

    这个线程的核心职责是按固定频率从队列中取动作并发送给机器人执行。
    它是机器人"身体"的驱动者。

    工作流程：
    ┌─────────────────────────────────────────────────────────┐
    │ LOOP（按 1/fps 的频率循环）：                            │
    │   action = action_queue.get()                           │
    │   if action 不为 None:                                  │
    │     ├─ 队列恢复后的混合（recovery blend）处理            │
    │     ├─ 安全检查（关节跳变 > 0.5 rad 则警告）            │
    │     └─ env.step(action) —— 发送动作给机器人              │
    │   else:                                                 │
    │     └─ 队列空，计数停顿（stall）                         │
    │   sleep 到下一个周期                                     │
    └─────────────────────────────────────────────────────────┘

    关键设计决策：
    - 固定频率循环：每 1/fps 秒尝试取一个动作。如果队列为空
      （Producer 还没产出），则"空转"一次并计数停顿。
    - Recovery blend：当队列从空变为非空时（从停顿中恢复），
      前几帧在旧动作和新动作之间做线性混合，避免动作跳变。
    - 安全检查：监控相邻帧之间的关节角度跳变，超过 0.5 弧度
      时打印警告。

    Args:
        env:                   机器人环境（用于执行动作）。
        action_queue:          RTC 动作队列。
        shutdown_event:        关闭信号。
        fps:                   控制频率（Hz）。
        record_data:           录制动作数据的列表（会被原地修改）。
        time_origin:           wall clock 时间原点。
        perf_origin:           性能计数器原点。
        recovery_blend_frames: 从停顿恢复时的混合帧数（默认 6）。
    """
    try:
        logger.info("[CONSUMER] Starting control consumer thread")

        # 每个周期的时长（秒）
        action_interval = 1.0 / fps

        last_action = None      # 上一步执行的最后一个动作
        step_count = 0          # 已执行步数
        stall_count = 0         # 连续停顿次数（队列为空的次数）
        current_source_action_index = None  # 当前动作在原始序列中的索引

        # Recovery blend 状态
        recovery_blend_total = max(0, int(recovery_blend_frames))
        recovery_blend_remaining = 0        # 剩余混合帧数
        recovery_hold_action = None         # 停顿前的最后一个动作（混合的起点）

        # =====================================================================
        # Consumer 主循环
        # =====================================================================
        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            queue_size_before = action_queue.qsize()
            source_action_index = current_source_action_index

            # ------------------------------------------------------------------
            # 从队列取一个动作
            # ------------------------------------------------------------------
            action_index_before_get = action_queue.get_action_index()
            action = action_queue.get()
            recovered_stall_count = 0

            if action is not None:
                # 队列非空 → 记录恢复信息
                recovered_stall_count = stall_count
                current_source_action_index = action_index_before_get
                source_action_index = current_source_action_index
            else:
                # 队列空 → 重置混合状态，计数停顿
                recovery_blend_remaining = 0
                recovery_hold_action = None
                stall_count += 1
                if stall_count % 50 == 1:
                    logger.warning(
                        "[CONSUMER] Queue empty (stalled %d times), step=%d",
                        stall_count,
                        step_count,
                    )

            if action is not None:
                # ------------------------------------------------------------------
                # Recovery Blend：从停顿中恢复时做平滑过渡
                #
                # 当 Consumer 因队列空而停顿后拿到第一个新动作时，
                # 新动作可能和机器人当前的关节位置有明显差异（跳变）。
                # Recovery blend 在前几帧做 last_action → new_action 的
                # 线性混合，避免突兀的关节运动。
                # ------------------------------------------------------------------
                if recovered_stall_count > 0 and last_action is not None:
                    recovery_blend_remaining = recovery_blend_total
                    recovery_hold_action = last_action.copy()
                    if recovery_blend_remaining > 0:
                        logger.warning(
                            "[CONSUMER] Recovered after %d empty ticks; "
                            "blending next %d frames",
                            recovered_stall_count,
                            recovery_blend_total,
                        )

                if recovery_blend_remaining > 0 and recovery_hold_action is not None:
                    blend_index = recovery_blend_total - recovery_blend_remaining + 1
                    # 线性混合：从 0 到 1
                    alpha = min(1.0, blend_index / max(1, recovery_blend_total))
                    action = (
                        (1.0 - alpha) * recovery_hold_action + alpha * action
                    ).astype(action.dtype, copy=False)
                    recovery_blend_remaining -= 1
                    if recovery_blend_remaining == 0:
                        recovery_hold_action = None

                # ------------------------------------------------------------------
                # 安全检查：相邻帧关节角度跳变监控
                # 排除夹爪维度（索引 7 和 15），因为夹爪开合通常变化较大
                # ------------------------------------------------------------------
                if last_action is not None:
                    # 提取双臂关节（不含夹爪）
                    arm_action = np.concatenate((action[:7], action[8:15]))
                    last_arm_action = np.concatenate(
                        (last_action[:7], last_action[8:15])
                    )
                    error = np.abs(arm_action - last_arm_action)
                    joint_id = int(np.argmax(error))
                    max_diff = float(error[joint_id])
                    if max_diff >= 0.5:
                        logger.warning(
                            "[CONSUMER] Large action jump: joint %d diff=%.4f "
                            "at step %d",
                            joint_id,
                            max_diff,
                            step_count,
                        )

                # ------------------------------------------------------------------
                # 执行动作
                # ------------------------------------------------------------------
                action_start_wall = time.time()
                action_start_perf = time.perf_counter()
                env.step(action)
                action_end_perf = time.perf_counter()
                action_end_wall = time.time()

                # 录制数据
                if record_data is not None:
                    record_data.append(
                        {
                            "step_index": step_count,
                            "queue_size": action_queue.qsize(),
                            "queue_size_before": queue_size_before,
                            "source_action_index": (
                                source_action_index
                                if source_action_index is not None
                                else float("nan")
                            ),
                            "action_start_time_s": action_start_wall
                            - (time_origin or action_start_wall),
                            "action_start_perf_s": action_start_perf
                            - (perf_origin or action_start_perf),
                            "action_end_time_s": action_end_wall
                            - (time_origin or action_end_wall),
                            "action_end_perf_s": action_end_perf
                            - (perf_origin or action_end_perf),
                            "step_duration_ms": (action_end_perf - action_start_perf)
                            * 1000.0,
                            "action": action.copy(),
                        }
                    )

                last_action = action
                step_count += 1
                stall_count = 0  # 重置停顿计数

            # ------------------------------------------------------------------
            # 频率控制：保证精确的 action_interval 间隔
            # 减去 1ms 的安全余量，防止因 sleep 精度问题导致周期略长
            # ------------------------------------------------------------------
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - elapsed - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[CONSUMER] Shutting down: %d steps, %d stalls", step_count, stall_count)
    except Exception:
        logger.error("[CONSUMER] Fatal exception:\n%s", traceback.format_exc())
        shutdown_event.set()  # 通知 Producer 也退出
        sys.exit(1)


# =============================================================================
# _save_records —— 保存录制数据到 CSV
# =============================================================================
def _save_records(
    config_profile: dict,
    record_states: list,
    record_actions: list,
) -> None:
    """将 Producer 和 Consumer 录制的时间序列数据保存为 CSV 文件。

    RTC 模式下的录制比同步模式更丰富：
    - record_states: 每次推理时的状态 + 大量元数据（时间戳、延迟、
      队列大小、merge 诊断等）
    - record_actions: 每次执行的 action + 元数据（步数、队列大小、
      来源索引等）

    输出的 CSV 带表头，方便在 Excel/Python/pandas 中直接分析。

    Args:
        config_profile: 部署配置字典。
        record_states:  状态记录列表。
        record_actions: 动作记录列表。
    """
    # 获取输出路径
    action_save_path, state_save_path = record_paths(
        config_profile,
        action_key="rtc_action_output_path",
        state_key="rtc_state_output_path",
        action_suffix="rtc_action_data",
        state_suffix="rtc_state_data",
    )

    # ------------------------------------------------------------------
    # 定义元数据字段列表（按 CSV 列顺序）
    # 这些字段提供了丰富的诊断信息，用于分析 RTC 系统的性能
    # ------------------------------------------------------------------

    # State 记录的元数据字段
    state_meta_fields = [
        "infer_index",                    # 推理序号
        "queue_size",                     # 队列大小
        "action_index_before",            # 合并前的动作索引
        "initial_delay",                  # 初始延迟步数
        "inference_delay_p95",            # P95 推理延迟估计
        "inference_delay",                # 当前推理延迟估计
        "measured_inference_delay",        # 实际测量延迟
        "used_delay",                     # 实际使用的延迟
        "inference_delay_minus_used",     # 延迟估计偏差
        "action_horizon",                 # 动作窗口 H
        "execution_horizon",              # 执行窗口 s
        "rtc_guidance_enabled",           # RTC guidance 是否启用
        "rtc_guidance_weight",            # RTC guidance 权重
        "action_postprocess_enabled",     # 后处理是否启用
        "action_postprocess_merge_delay", # 后处理的 merge 延迟
        "action_postprocess_blend_frames",# 混合帧数
        "action_postprocess_ema_frames",  # EMA 帧数
        "action_postprocess_mae",         # 后处理 MAE（平均绝对误差）
        "action_postprocess_max",         # 后处理最大差异
        "trigger_queue_size",             # 触发队列阈值
        "paper_s",                        # 论文参数 s（需新补的步数）
        "prefix_horizon",                 # 旧动作前缀长度
        "prev_left_over_len",             # 上一批剩余长度
        "obs_request_perf_s",             # 观测请求时间
        "obs_receive_time_s",             # 观测接收时间
        "obs_receive_perf_s",             # 观测接收 perf 时间
        "obs_wait_ms",                    # 观测等待耗时
        "obs_bridge_ref_timestamp_ms",    # 观测参考时间戳
        "obs_joint_timestamp_ms",         # 关节时间戳
        "obs_gripper_timestamp_ms",       # 夹爪时间戳
        "obs_image_timestamp_ms",         # 图像时间戳
        "obs_cam_high_timestamp_ms",      # 高位相机时间戳
        "obs_cam_left_wrist_timestamp_ms",# 左腕相机时间戳
        "obs_cam_right_wrist_timestamp_ms",# 右腕相机时间戳
        "obs_age_ms",                     # 观测总时长
        "obs_joint_age_ms",               # 关节数据时长
        "obs_gripper_age_ms",             # 夹爪数据时长
        "obs_image_age_ms",               # 图像数据时长
        "obs_sensor_time_s",              # 传感器时间
        "obs_joint_sensor_time_s",        # 关节传感器时间
        "obs_joint_ref_offset_ms",        # 关节-参考时间偏移
        "obs_gripper_ref_offset_ms",      # 夹爪-参考时间偏移
        "obs_image_span_ms",              # 图像时间跨度
        # ---- Merge 诊断 ----
        "merge_count",                    # 合并次数
        "merge_used_delay",               # 合并使用的延迟
        "merge_extra_delay",              # 合并额外延迟
        "merge_pre_qsize",                # 合并前队列大小
        "merge_pre_index",                # 合并前索引
        # ---- Boundary 诊断（processed 动作） ----
        "boundary_new_index",             # 新动作边界索引
        "boundary_proc_plan_mae",         # 处理后计划 MAE
        "boundary_proc_plan_max",         # 处理后计划最大差异
        "boundary_proc_plan_max_dim",     # 最大差异所在维度
        "boundary_proc_plan_max_delta",   # 最大差异值
        "boundary_proc_exec_mae",         # 处理后执行 MAE
        "boundary_proc_exec_max",         # 处理后执行最大差异
        "boundary_proc_exec_max_dim",     # 最大差异维度
        "boundary_proc_exec_max_delta",   # 最大差异值
        # ---- Boundary 诊断（仅机械臂） ----
        "boundary_proc_plan_arm_mae",     # 处理后计划 臂部 MAE
        "boundary_proc_plan_arm_max",     # 处理后计划 臂部最大
        "boundary_proc_plan_arm_max_dim",
        "boundary_proc_plan_arm_max_delta",
        "boundary_proc_exec_arm_mae",
        "boundary_proc_exec_arm_max",
        "boundary_proc_exec_arm_max_dim",
        "boundary_proc_exec_arm_max_delta",
        # ---- Boundary 诊断（仅夹爪） ----
        "boundary_proc_plan_gripper_mae",
        "boundary_proc_plan_gripper_max",
        "boundary_proc_plan_gripper_max_dim",
        "boundary_proc_plan_gripper_max_delta",
        "boundary_proc_exec_gripper_mae",
        "boundary_proc_exec_gripper_max",
        "boundary_proc_exec_gripper_max_dim",
        "boundary_proc_exec_gripper_max_delta",
        # ---- Boundary 诊断（原始动作） ----
        "boundary_raw_plan_mae",
        "boundary_raw_plan_max",
        "boundary_raw_plan_max_dim",
        "boundary_raw_plan_max_delta",
    ]

    # Action 记录的元数据字段
    action_meta_fields = [
        "step_index",
        "queue_size",
        "queue_size_before",
        "source_action_index",
        "action_start_time_s",
        "action_start_perf_s",
        "action_end_time_s",
        "action_end_perf_s",
        "step_duration_ms",
    ]

    # ------------------------------------------------------------------
    # 构建 CSV 表头
    # 格式：元数据字段 + L_arm_0 ... L_arm_6 + L_grip + R_arm_0 ... R_arm_6 + R_grip
    # ------------------------------------------------------------------
    state_header = ",".join(
        state_meta_fields
        + [f"L_arm_{i}" for i in range(7)]
        + ["L_grip"]
        + [f"R_arm_{i}" for i in range(7)]
        + ["R_grip"]
    )
    action_header = ",".join(
        action_meta_fields
        + [f"L_arm_{i}" for i in range(7)]
        + ["L_grip"]
        + [f"R_arm_{i}" for i in range(7)]
        + ["R_grip"]
    )

    # ------------------------------------------------------------------
    # 写入 State 数据
    # ------------------------------------------------------------------
    if record_states:
        state_array = np.vstack(
            [
                np.concatenate(
                    (
                        # 元数据列
                        np.array(
                            [record.get(field, float("nan")) for field in state_meta_fields],
                            dtype=np.float64,
                        ),
                        # State 数据列（关节角度等）
                        record["state"],
                    )
                )
                for record in record_states
            ]
        )
        np.savetxt(
            state_save_path, state_array,
            delimiter=",", fmt="%.6f", header=state_header, comments="",
        )
        logger.info(
            "Saved %d states to %s (shape=%s)",
            len(record_states), state_save_path, state_array.shape,
        )

    # ------------------------------------------------------------------
    # 写入 Action 数据
    # ------------------------------------------------------------------
    if record_actions:
        action_array = np.vstack(
            [
                np.concatenate(
                    (
                        np.array(
                            [record.get(field, float("nan")) for field in action_meta_fields],
                            dtype=np.float64,
                        ),
                        record["action"],
                    )
                )
                for record in record_actions
            ]
        )
        np.savetxt(
            action_save_path, action_array,
            delimiter=",", fmt="%.6f", header=action_header, comments="",
        )
        logger.info(
            "Saved %d actions to %s (shape=%s)",
            len(record_actions), action_save_path, action_array.shape,
        )


# =============================================================================
# main —— 主函数
# =============================================================================
def main() -> None:
    """TRON2 RTC 部署的主入口。

    与同步客户端 pi_client.py 不同，main() 不直接执行控制循环，
    而是负责：
    1. 加载和验证 RTC 配置
    2. 初始化机器人环境和策略连接
    3. 执行预热推理
    4. 启动 Producer 和 Consumer 两个后台线程
    5. 监控运行状态，处理优雅退出

    控制流程：
    ┌──────────────────────────────────────────────────────────┐
    │  1. 加载配置（YAML + 命令行参数）                          │
    │  2. 验证服务器支持 RTC（通过 metadata 检查）               │
    │  3. 初始化环境、策略客户端、ActionQueue                    │
    │  4. 执行 warmup（预热推理 + 填充队列）                     │
    │  5. 启动 Producer 线程（daemon）                          │
    │  6. 启动 Consumer 线程（daemon）                          │
    │  7. 主线程循环监控（每 5 秒打印状态）                       │
    │  8. 退出时：设置 shutdown_event → join 线程 → 保存录制数据  │
    └──────────────────────────────────────────────────────────┘
    """
    # ------------------------------------------------------------------
    # 第一步：加载配置
    # ------------------------------------------------------------------
    args = _parse_args()
    profile_path = select_profile_path(args.profile, args.deploy_config)
    config_profile = load_deploy_config(profile_path)
    client_profile = section(config_profile, "client")

    # 安全检查：RTC 客户端必须启用 rtc_enabled
    if not bool_value(client_profile.get("rtc_enabled", False)):
        raise ValueError(
            "client.rtc_enabled is false or missing. "
            "Use examples/tron2/pi_client.py for synchronous inference."
        )

    # ------------------------------------------------------------------
    # 配置日志级别
    # RTC 模式下有大量日志输出，可调整级别来控制详细程度：
    #   DEBUG: 最详细（用于调试）
    #   INFO:  正常（推荐）
    #   WARNING: 仅警告和错误
    # 另外单独提高 ActionQueue 的日志级别，避免过于嘈杂的 merge 日志
    # ------------------------------------------------------------------
    log_level = str(client_profile.get("log_level", "INFO")).upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger("tron2_env.rtc.action_queue").setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # 第二步：提取配置参数
    # ------------------------------------------------------------------
    env_config = build_env_config(config_profile)

    # fps: 控制频率（Hz）
    fps = float(client_profile.get("fps", env_config.fps))

    # RTC 模式选择：
    # - rtc_guidance_enabled=True:  客户端侧 guidance 加权融合
    # - trained_rtc_mode=True:      策略模型内部处理 RTC（无需客户端 guidance）
    # - 两者都为 False:             replace-only 模式（纯替换，不做融合）
    rtc_guidance_enabled = bool_value(client_profile.get("rtc_guidance_enabled", True))
    rtc_guidance_weight = _resolve_guidance_weight(client_profile)
    trained_rtc_mode = bool_value(client_profile.get("trained_rtc_mode", False))

    # 观测获取超时总预算（秒）
    obs_timeout_budget_s = float(client_profile.get("obs_timeout_budget_s", 5.0))

    # 动作后处理器（客户端平滑）
    action_postprocessor = _resolve_action_postprocess(client_profile)

    # ------------------------------------------------------------------
    # 第三步：初始化基础设施
    # ------------------------------------------------------------------
    latency_stats = LatencyTracker()  # 推理延迟统计
    shutdown_event = Event()           # 线程间关闭信号

    # Prompt 控制器（支持运行时通过 stdin 修改任务指令）
    prompt_ctrl = PromptController(args.prompt or client_profile.get("prompt"))
    prompt_ctrl.start_stdin_listener()

    # 录制数据缓冲区
    record_states: list = []
    record_actions: list = []

    logger.info("Observation source: %s", env_config.observation_source)
    logger.info("Control backend: %s", env_config.control_backend)

    # ------------------------------------------------------------------
    # 第四步：连接机器人环境 & 策略服务器
    # ------------------------------------------------------------------
    with Tron2Env(env_config) as env:
        env.reset()
        logger.info("Robot initialized. Connecting to policy server...")

        # 创建 WebSocket 连接到策略服务器
        ws_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host(client_profile),
            port=policy_port(client_profile),
        )

        # ------------------------------------------------------------------
        # 验证服务器端的 RTC 支持
        # get_server_metadata() 返回服务器能力信息，包括：
        #   - rtc_enabled: 是否支持 RTC 推理
        #   - action_horizon: 动作窗口长度 H
        # ------------------------------------------------------------------
        server_meta = ws_client.get_server_metadata()
        rtc_enabled = bool(server_meta.get("rtc_enabled", False))
        if not rtc_enabled:
            raise RuntimeError(
                "The policy server does not report rtc_enabled=True. "
                "Use pi_client.py for non-RTC inference "
                "or start a server with an RTC-capable model."
            )

        # 从服务器获取动作窗口长度 H
        action_horizon = int(server_meta.get("action_horizon", DEFAULT_ACTION_HORIZON))

        # 解析 RTC 时序参数
        execution_horizon, delay, trigger_queue_size = _resolve_rtc_timing(
            client_profile, action_horizon
        )

        # 打印配置摘要
        logger.info("Server action_horizon=%d", action_horizon)
        logger.info(
            "Config: fps=%.1f, H=%d, s=%d, init_d=%d, trigger_queue_size=H-s=%d, "
            "guide=%s, weight=%.2f, trained_rtc=%s, post=%s",
            fps,
            action_horizon,
            execution_horizon,
            delay,
            trigger_queue_size,
            rtc_guidance_enabled,
            rtc_guidance_weight,
            trained_rtc_mode,
            action_postprocessor.describe(),
        )

        # ------------------------------------------------------------------
        # 第五步：创建 ActionQueue 并设置时间原点
        # ActionQueue 是 Producer 和 Consumer 之间的线程安全桥梁
        # ------------------------------------------------------------------
        action_queue = ActionQueue(rtc_enabled=rtc_enabled)

        # 设置时间原点，用于计算相对时间戳（便于多线程时间对齐）
        time_origin = time.time()
        perf_origin = time.perf_counter()

        # ------------------------------------------------------------------
        # 第六步：预热推理
        # 1. 做一次非 RTC 推理（验证通信）
        # 2. 做一次 RTC 推理（验证 RTC 路径）
        # 3. 将预热结果填充到队列中（Consumer 启动后立即可用）
        # ------------------------------------------------------------------
        warmup_result = warmup_rtc(
            ws_client,
            env,
            prompt_ctrl,
            rtc_guidance_enabled,
            rtc_guidance_weight,
            delay,
            action_horizon=action_horizon,
            trained_rtc_mode=trained_rtc_mode,
        )

        # 用预热结果填充队列
        warmup_raw = warmup_result["raw_actions"]
        warmup_proc = np.stack(warmup_result["actions"], axis=0)
        action_queue.merge(warmup_raw, warmup_proc, real_delay=0)

        # 记录预热延迟到统计器
        latency_stats.reset()
        warmup_latency_s = float(warmup_result.get("client_warmup_latency_s") or 0.0)
        if warmup_latency_s > 0:
            latency_stats.add(warmup_latency_s)
        logger.info(
            "[INIT] Queue seeded with %d warmup actions, latency seed %.1fms/%d frames",
            action_queue.qsize(),
            warmup_latency_s * 1000.0,
            math.ceil(warmup_latency_s * fps) if warmup_latency_s > 0 else 0,
        )

        # ------------------------------------------------------------------
        # 第七步：启动双线程
        # ------------------------------------------------------------------

        # Producer 线程：负责"观测 → 推理 → merge"
        producer_thread = Thread(
            target=inference_producer,
            args=(
                ws_client,
                env,
                action_queue,
                latency_stats,
                shutdown_event,
                fps,
                execution_horizon,
                delay,
                rtc_guidance_enabled,
                rtc_guidance_weight,
                trigger_queue_size,
                action_postprocessor,
            ),
            kwargs={
                "action_horizon": action_horizon,
                "record_states": record_states,
                "time_origin": time_origin,
                "perf_origin": perf_origin,
                "trained_rtc_mode": trained_rtc_mode,
                "prompt_controller": prompt_ctrl,
                "obs_timeout_budget_s": obs_timeout_budget_s,
            },
            daemon=True,   # daemon 线程：主线程退出时自动终止
            name="Producer",
        )
        producer_thread.start()

        # Consumer 线程：负责"取动作 → 执行"
        consumer_thread = Thread(
            target=control_consumer,
            args=(env, action_queue, shutdown_event, fps),
            kwargs={
                "record_data": record_actions,
                "time_origin": time_origin,
                "perf_origin": perf_origin,
                "recovery_blend_frames": int(
                    client_profile.get("obs_recovery_blend_frames", 6)
                ),
            },
            daemon=True,
            name="Consumer",
        )
        consumer_thread.start()

        # ------------------------------------------------------------------
        # 第八步：主线程监控循环
        # 主线程不再参与控制，而是作为"看门狗"监控两个工作线程的状态
        # ------------------------------------------------------------------
        duration = float(client_profile.get("duration", 120.0))
        if duration and duration > 0:
            logger.info("Running for %.0f seconds...", duration)
        else:
            logger.info("Running indefinitely (duration=0). Ctrl+C to stop.")
        start_time = time.time()

        try:
            while not shutdown_event.is_set() and (
                duration <= 0 or (time.time() - start_time) < duration
            ):
                # 每 5 秒打印一次状态摘要
                time.sleep(5)
                logger.info(
                    "[MAIN] queue=%d, action_idx=%d, latency_p95=%.1fms",
                    action_queue.qsize(),
                    action_queue.get_action_index(),
                    (latency_stats.p95() or 0) * 1000,
                )
        except KeyboardInterrupt:
            # 用户按 Ctrl+C 优雅退出
            logger.info("Interrupted by operator.")
        finally:
            # ------------------------------------------------------------------
            # 第九步：清理和退出
            # ------------------------------------------------------------------
            logger.info("Stopping RTC threads")
            shutdown_event.set()           # 通知两个线程停止

            # join(timeout=5): 等待线程在 5 秒内退出
            producer_thread.join(timeout=5)
            consumer_thread.join(timeout=5)

            # 保存录制的数据
            _save_records(config_profile, record_states, record_actions)

    logger.info("Cleanup completed")


# =============================================================================
# 脚本入口
# =============================================================================
if __name__ == "__main__":
    # 配置全局 logging 格式
    # 格式：时间 [模块名] 级别: 消息
    # force=True 确保覆盖任何已有的 logging 配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )
    main()
