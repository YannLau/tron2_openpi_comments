"""Public TRON2 deployment YAML helpers.

=============================================================================
背景 / Background
=============================================================================
本模块是 TRON2 示例与 OpenPI 部署基础设施之间的"胶水层"（glue layer）。
它完成三件事情：

  1. 加载和解析部署 YAML 配置文件
     - 配置文件包含 robot（机器人）、camera（相机）、bridge（桥接）、
       client（客户端）、policy（策略）等子配置块

  2. 将 YAML 配置转换为 TRON2 所需的强类型配置对象
     - Tron2Config、CameraConfig、BridgeConfig、EnvConfig 等
     - 处理默认值、兼容旧字段名、归一化相机名称等

  3. 提供运行时辅助函数
     - 格式化观测数据（format_obs）供推理服务器使用
     - 时间戳计算（timestamp_ms, age_ms, relative_sensor_time_s）
     - 线程安全的 prompt 控制器（PromptController）
     - 推理性能计时（infer_with_timing）

为什么需要这个模块？
  示例代码（如 run_serving.py、teleop_and_record.py）通常只有几十行，
  它们依赖本模块处理所有繁琐的配置解析和类型转换，保持示例干净简洁。

与兄弟项目 tron2_env 的关系：
  本模块通过 _external_tron2_env.py 将 ../tron2_env/src 加入 sys.path，
  然后从中导入 Tron2Config、CameraConfig 等配置类。
=============================================================================
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# einops：爱因斯坦求和约定的张量操作库
#     这里用于 rearrange(图像, "h w c -> c h w")，即把通道维从最后一维移到第一维
#     （例如 PIL/OpenCV 的 HWC 格式 → PyTorch 的 CHW 格式）
import einops
# NumPy：Python 科学计算基础库，这里用于处理图像数组和关节角度数组
import numpy as np

# OpenPI 内部的部署配置加载器（私有模块）
# _deploy_config 提供了 load_deploy_config()、section()、bool_value() 等
# 通用 YAML 配置加载函数
from openpi.shared import deploy_config as _deploy_config
# image_tools：图像处理工具，提供 resize_with_pad（缩放+填充）和 convert_to_uint8（类型转换）
from openpi_client import image_tools

# ---------------------------------------------------------------------------
# 将兄弟项目 tron2_env/src 加入 sys.path，使得下面的 import tron2_env 可用
# 详见 _external_tron2_env.py 的注释
# ---------------------------------------------------------------------------
from _external_tron2_env import ensure_external_tron2_env_on_path

ensure_external_tron2_env_on_path()

# 从 tron2_env 包中导入配置数据类
#    BridgeConfig  — WebSocket 桥接配置（主机地址、图像/关节 topic 等）
#    CameraConfig  — 相机配置（分辨率、队列大小、调试图像等）
#    EnvConfig     — 环境总配置（组合 robot + camera + bridge + client）
#    Tron2Config   — 机器人配置（IP、端口、初始关节角等）
from tron2_env import BridgeConfig, CameraConfig, EnvConfig, Tron2Config

# ---------------------------------------------------------------------------
# 日志记录器：使用 Python 标准库 logging
# __name__ 是当前模块的完整限定名（"examples.tron2.deploy_config" 或 "__main__"）
# 这样用户可以在日志中看到日志来源模块
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ============================================================================
# 常量 / Constants
# ============================================================================

# ---------------------------------------------------------------------------
# DEFAULT_INIT_JOINTS：机器人的默认初始关节角度（14 个关节）
# ---------------------------------------------------------------------------
# TRON2 机器人有 14 个自由度（双臂各 7 个关节）。
# 这里的 14 个浮点数指定了机器人启动时的默认关节角度（单位：弧度 rad）。
# 在 YAML 配置中可以通过 robot.init_joints 覆盖。
#
# 为什么需要初始关节角度？
#   机器人启动时需要知道"应该先走到哪个姿态"。
#   如果配置中没有指定，就使用这里的默认值。
#   这些值是经过实际调试后选定的安全初始位姿。
# ---------------------------------------------------------------------------
DEFAULT_INIT_JOINTS = [
    0.026899,       # 关节 1
    0.2612,         # 关节 2
    -0.02709991,    # 关节 3
    -1.5477003,     # 关节 4
    0.265,          # 关节 5
    0.0180999,      # 关节 6
    -0.0614999,     # 关节 7
    0.008999,       # 关节 8
    -0.269,         # 关节 9
    0.02069998,     # 关节 10
    -1.5567001,     # 关节 11
    -0.254,         # 关节 12
    -0.02309972,    # 关节 13
    0.06469989,     # 关节 14
]

# ---------------------------------------------------------------------------
# OPENPI_CAMERA_NAMES：OpenPI 统一使用的三个相机名称
# ---------------------------------------------------------------------------
# TRON2 的观测系统有三个相机：
#   cam_high        — 高位俯视相机（头部相机）
#   cam_left_wrist  — 左手腕相机
#   cam_right_wrist — 右手腕相机
# ---------------------------------------------------------------------------
OPENPI_CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# ---------------------------------------------------------------------------
# LEGACY_CAMERA_NAME_MAP：旧版相机名 → 新版相机名的映射
# ---------------------------------------------------------------------------
# 旧版代码或旧配置文件中可能使用不同的相机命名：
#   head_camera_image → cam_high
#   left_wrist_image  → cam_left_wrist
#   right_wrist_image → cam_right_wrist
#
# 通过这个映射表，旧名称会被自动转换为统一的新名称，
# 确保向后兼容（backward compatibility）。
# ---------------------------------------------------------------------------
LEGACY_CAMERA_NAME_MAP = {
    "head_camera_image": "cam_high",
    "left_wrist_image": "cam_left_wrist",
    "right_wrist_image": "cam_right_wrist",
}


# ============================================================================
# YAML 配置加载 / Configuration Loading
# ============================================================================

def load_deploy_config(path: str | Path | None) -> dict[str, Any]:
    """加载部署 YAML 配置文件，返回嵌套的字典结构。

    这是对 openpi.shared.deploy_config.load_deploy_config 的薄封装。

    参数:
        path: YAML 文件路径（字符串或 Path 对象），或 None（使用默认路径）

    返回:
        dict[str, Any]: 解析后的配置字典，包含 robot、camera、bridge、
                         client、policy 等子字典

    示例配置文件的结构大致如下：
        robot:
          ip: "192.168.1.100"
          port: 5000
          init_joints: [0.0, 0.26, ...]
          init_head: 0.0
        camera:
          camera_names: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
          resolution: [480, 640, 3]
        bridge:
          host: "wss://bridge.example.com"
          ws_path: "/bridge/ws"
        client:
          fps: 30.0
          policy_host: "127.0.0.1"
          policy_port: 8000
          task: "pick_and_place"
    """
    return _deploy_config.load_deploy_config(path)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """从配置字典中提取指定名称的子配置块（section）。

    这是对 openpi.shared.deploy_config.section 的薄封装。

    参数:
        config: 完整的配置字典
        name:   section 名称（如 "robot", "camera", "bridge", "client"）

    返回:
        dict[str, Any]: 该 section 的配置字典。
                        如果 config 中没有该 section，返回空字典 {}。

    示例:
        >>> cfg = {"robot": {"ip": "192.168.1.100"}}
        >>> section(cfg, "robot")
        {'ip': '192.168.1.100'}
        >>> section(cfg, "missing_section")
        {}
    """
    return _deploy_config.section(config, name)


def bool_value(value: Any) -> bool:
    """将 YAML 中的各种"布尔"写法统一转换为 Python bool。

    这是对 openpi.shared.deploy_config.bool_value 的薄封装。

    YAML 中布尔值可以写成 true/false、yes/no、on/off、"True"/"False" 等。
    此外，数字 0/1 也会被正确地映射。

    参数:
        value: 从 YAML 读取的任意值

    返回:
        bool: 标准 Python 布尔值

    示例:
        >>> bool_value("yes")
        True
        >>> bool_value(0)
        False
    """
    return _deploy_config.bool_value(value)


# ============================================================================
# 相机名称归一化 / Camera Name Normalization
# ============================================================================

def _camera_name(name: str) -> str:
    """将单个相机名称从旧版命名转换为新版命名。

    内部辅助函数，通过 LEGACY_CAMERA_NAME_MAP 查表转换。
    如果名称不在映射表中，说明已经是新版名称，原样返回。

    参数:
        name: 相机名称（可能是旧版或新版）

    返回:
        str: 统一的新版相机名称

    示例:
        >>> _camera_name("head_camera_image")
        "cam_high"
        >>> _camera_name("cam_high")
        "cam_high"
    """
    return LEGACY_CAMERA_NAME_MAP.get(str(name), str(name))


def normalize_camera_profile(camera_profile: dict[str, Any]) -> dict[str, Any]:
    """归一化相机配置中的相机名称，确保所有名称都是新版格式。

    这个函数处理相机配置中的两个关键字段：
      1. serial_to_name — 相机序列号到名称的映射字典
      2. camera_names    — 相机名称列表

    归一化逻辑（按优先级）：
      1. 如果 camera_names 已经显式指定 → 逐项转换名称
      2. 否则如果 serial_to_name 存在 → 从 serial_to_name 的值中提取名称列表
      3. 否则 → 使用默认的 OPENPI_CAMERA_NAMES

    参数:
        camera_profile: 相机配置字典（来自 YAML 的 camera section）

    返回:
        dict[str, Any]: 归一化后的相机配置字典（不会修改原始输入）

    示例:
        >>> normalize_camera_profile({"camera_names": ["head_camera_image"]})
        {"camera_names": ["cam_high"]}
    """
    # 复制一份，避免修改调用方持有的原始字典
    profile = dict(camera_profile)
    serial_to_name = profile.get("serial_to_name")
    if isinstance(serial_to_name, dict):
        # 将 serial_to_name 中所有的值（相机名称）转换为新版名称
        profile["serial_to_name"] = {
            str(serial): _camera_name(name) for serial, name in serial_to_name.items()
        }

    camera_names = profile.get("camera_names")
    if camera_names:
        # 情况 1：camera_names 已显式指定 → 逐项转换
        profile["camera_names"] = [_camera_name(name) for name in camera_names]
    elif isinstance(serial_to_name, dict):
        # 情况 2：camera_names 未指定但 serial_to_name 存在 → 从 serial_to_name 的值中提取
        profile["camera_names"] = list(profile["serial_to_name"].values())
    else:
        # 情况 3：都没有 → 使用默认相机列表
        profile["camera_names"] = list(OPENPI_CAMERA_NAMES)

    return profile


# ============================================================================
# 配置对象构建 / Configuration Object Builders
# ============================================================================

def normalized_raw_config(config_profile: dict[str, Any]) -> dict[str, Any]:
    """生成归一化后的原始配置字典（用于 EnvConfig.raw_config 字段）。

    将 camera section 归一化处理后，嵌入到一个新的配置字典中。
    这使得原始配置字典中的相机名称与运行时使用的名称保持一致。

    参数:
        config_profile: 完整的配置字典

    返回:
        dict[str, Any]: 归一化后的原始配置字典
    """
    raw_config = dict(config_profile)
    raw_config["camera"] = normalize_camera_profile(section(config_profile, "camera"))
    return raw_config


def build_robot_config(config_profile: dict[str, Any]) -> Tron2Config:
    """从 YAML 配置构建 Tron2Config（机器人配置）对象。

    从配置字典的 robot section 中提取以下字段，并填充默认值：

    | 字段              | 默认值            | 说明                         |
    |-------------------|-------------------|------------------------------|
    | robot_ip          | "ROBOT_IP"        | 机器人控制器 IP 地址          |
    | port              | 5000              | 机器人控制端口                |
    | init_joints       | DEFAULT_INIT_JOINTS | 初始关节角（14 个弧度值）   |
    | init_head         | None              | 初始头部角度                  |
    | state_queue_maxlen| 7                 | 状态队列最大长度              |
    | polling_rate      | 200.0             | 状态轮询频率（Hz）            |
    | connection_timeout| 5.0               | 连接超时（秒）                |

    参数:
        config_profile: 完整的配置字典

    返回:
        Tron2Config: 强类型的机器人配置对象
    """
    robot_profile = section(config_profile, "robot")

    # 从配置中取 init_joints，如果没指定就用默认值
    init_joints = robot_profile.get("init_joints") or DEFAULT_INIT_JOINTS

    return Tron2Config(
        robot_ip=str(robot_profile.get("ip", "ROBOT_IP")),
        port=int(robot_profile.get("port", 5000)),
        init_joints=init_joints,
        init_head=robot_profile.get("init_head"),
        # 状态队列：机器人状态以流的方式到达，队列用于缓冲最近的 N 个状态
        state_queue_maxlen=int(robot_profile.get("state_queue_maxlen", 7)),
        # 轮询频率：每秒向机器人请求状态的次数
        polling_rate=float(robot_profile.get("polling_rate", 200.0)),
        # 连接超时：如果超过此时间还没连上机器人，则报错
        connection_timeout=float(robot_profile.get("connection_timeout", 5.0)),
    )


def build_camera_config(config_profile: dict[str, Any]) -> CameraConfig:
    """从 YAML 配置构建 CameraConfig（相机配置）对象。

    先调用 normalize_camera_profile 归一化相机名称，再构建配置对象。

    | 字段             | 默认值               | 说明                        |
    |------------------|----------------------|-----------------------------|
    | camera_names     | OPENPI_CAMERA_NAMES  | 三个相机的名称列表           |
    | resolution       | [480, 640, 3]        | 图像分辨率（H, W, C）       |
    | max_queue_size   | 10                   | 图像队列最大长度             |
    | save_debug_images| False                | 是否保存调试图像             |
    | debug_image_dir  | "debug_images"       | 调试图像的保存目录           |

    参数:
        config_profile: 完整的配置字典

    返回:
        CameraConfig: 强类型的相机配置对象
    """
    # 先归一化相机名称（兼容旧版命名）
    camera_profile = normalize_camera_profile(section(config_profile, "camera"))
    resolution = camera_profile.get("resolution", [480, 640, 3])

    return CameraConfig(
        camera_names=list(camera_profile.get("camera_names", OPENPI_CAMERA_NAMES)),
        resolution=tuple(resolution),  # 确保 resolution 是元组 (H, W, C)
        max_queue_size=int(camera_profile.get("max_queue_size", 10)),
        save_debug_images=bool_value(camera_profile.get("save_debug_images", False)),
        debug_image_dir=str(camera_profile.get("debug_image_dir", "debug_images")),
    )


def build_bridge_config(config_profile: dict[str, Any]) -> BridgeConfig:
    """从 YAML 配置构建 BridgeConfig（WebSocket 桥接配置）对象。

    Bridge 是 TRON2 架构中的消息中转服务，它通过 WebSocket 在：
      - 机器人控制器
      - 相机服务器
      - 推理策略服务器
      - 客户端
    之间传递图像帧和关节状态数据。

    | 字段              | 默认值                      | 说明                       |
    |-------------------|-----------------------------|----------------------------|
    | host              | "wss://BRIDGE_HOST"         | Bridge 服务 WebSocket 地址  |
    | ws_path           | "/bridge/ws"                | WebSocket 路径              |
    | image_max_fps     | 0（不限制）                  | 图像最大帧率                |
    | align_max_delay_ms| 200                         | 多模态对齐最大延迟（ms）     |
    | verify_tls        | False                       | 是否验证 TLS 证书           |
    | image_topics      | BridgeConfig()的默认值       | 图像数据的 topic 映射       |
    | joint_topics      | BridgeConfig()的默认值       | 关节数据的 topic 映射       |
    | save_debug_images | False                       | 是否保存调试图像            |
    | debug_image_dir   | "debug_images"              | 调试图像的保存目录          |

    参数:
        config_profile: 完整的配置字典

    返回:
        BridgeConfig: 强类型的 Bridge 配置对象
    """
    bridge_profile = section(config_profile, "bridge")

    return BridgeConfig(
        host=str(bridge_profile.get("host", "wss://BRIDGE_HOST")),
        ws_path=str(bridge_profile.get("ws_path", "/bridge/ws")),
        # image_max_fps=0 表示不限制帧率
        image_max_fps=int(bridge_profile.get("image_max_fps", 0)),
        # align_max_delay_ms：多模态数据（图像+关节）对齐时允许的最大时间差
        align_max_delay_ms=int(bridge_profile.get("align_max_delay_ms", 200)),
        verify_tls=bool_value(bridge_profile.get("verify_tls", False)),
        # 如果没有指定 topic 映射，使用 BridgeConfig 实例的默认值
        image_topics=dict(bridge_profile.get("image_topics", BridgeConfig().image_topics)),
        joint_topics=dict(bridge_profile.get("joint_topics", BridgeConfig().joint_topics)),
        save_debug_images=bool_value(bridge_profile.get("save_debug_images", False)),
        debug_image_dir=str(bridge_profile.get("debug_image_dir", "debug_images")),
    )


def build_env_config(config_profile: dict[str, Any]) -> EnvConfig:
    """从 YAML 配置构建 EnvConfig（环境总配置）对象。

    EnvConfig 是整个 TRON2 环境的顶层配置，它组合了：
      - Tron2Config（机器人）
      - CameraConfig（相机）
      - BridgeConfig（桥接）
      - 以及客户端相关的参数

    从 client section 和 robot section 中提取的客户端参数：

    | 字段                   | 默认值      | 说明                           |
    |------------------------|-------------|--------------------------------|
    | control_backend        | "websocket" | 控制后端类型                    |
    | publish_rate           | 300.0       | 动作发布频率（Hz）               |
    | fps                    | 30.0        | 客户端帧率                      |
    | time_sync_tolerance    | 0.01        | 时间同步容差（秒）               |
    | time_sync_max_retries  | 3           | 时间同步最大重试次数              |
    | legacy_use_time_sync   | True        | 是否使用旧版时间同步             |
    | state_dim              | 16          | 状态向量维度                     |
    | observation_source     | "legacy"    | 观测数据来源                     |
    | bridge_state_source    | "bridge"    | Bridge 状态来源                  |

    参数:
        config_profile: 完整的配置字典

    返回:
        EnvConfig: 强类型的环境配置对象
    """
    client_profile = section(config_profile, "client")
    robot_profile = section(config_profile, "robot")
    bridge_profile = section(config_profile, "bridge")

    fps = float(client_profile.get("fps", 30.0))
    return EnvConfig(
        robot_config=build_robot_config(config_profile),
        camera_config=build_camera_config(config_profile),
        # control_backend 优先使用 client 中的设置，否则回退到 robot 中的设置
        # 两种都未设置时默认 "websocket"
        control_backend=str(
            client_profile.get("control_backend", robot_profile.get("control_backend", "websocket"))
        ),
        # publish_rate 同理：client 优先，robot 兜底
        publish_rate=float(client_profile.get("publish_rate", robot_profile.get("publish_rate", 300.0))),
        fps=fps,
        time_sync_tolerance=float(client_profile.get("time_sync_tolerance", 0.01)),
        time_sync_max_retries=int(client_profile.get("time_sync_max_retries", 3)),
        legacy_use_time_sync=bool_value(client_profile.get("legacy_use_time_sync", True)),
        state_dim=int(client_profile.get("state_dim", 16)),
        observation_source=str(client_profile.get("observation_source", "legacy")),
        # bridge_state_source 优先 bridge 中的 setting，再回退到 client
        bridge_state_source=str(
            bridge_profile.get("state_source", client_profile.get("bridge_state_source", "bridge"))
        ),
        bridge_config=build_bridge_config(config_profile),
        raw_config=normalized_raw_config(config_profile),
    )


# ============================================================================
# 策略服务地址 / Policy Server Address Helpers
# ============================================================================

def policy_host(client_profile: dict[str, Any]) -> str:
    """获取推理策略服务器的 IP 地址或主机名。

    从 client section 中按以下优先级查找：
      1. policy_host（新字段名）
      2. server_host（旧字段名——向后兼容）
      3. "127.0.0.1"（默认本地地址）

    参数:
        client_profile: client section 的配置字典

    返回:
        str: 策略服务器地址

    示例:
        >>> policy_host({"policy_host": "10.0.0.5"})
        "10.0.0.5"
        >>> policy_host({"server_host": "192.168.1.1"})
        "192.168.1.1"
        >>> policy_host({})
        "127.0.0.1"
    """
    return str(client_profile.get("policy_host", client_profile.get("server_host", "127.0.0.1")))


def policy_port(client_profile: dict[str, Any]) -> int:
    """获取推理策略服务器的端口号。

    从 client section 中按以下优先级查找：
      1. policy_port（新字段名）
      2. server_port（旧字段名——向后兼容）
      3. port（更旧的通用字段名——向后兼容）
      4. 8000（默认端口）

    参数:
        client_profile: client section 的配置字典

    返回:
        int: 策略服务器端口号

    示例:
        >>> policy_port({"policy_port": 9000})
        9000
        >>> policy_port({"port": 8080})
        8080
        >>> policy_port({})
        8000
    """
    return int(client_profile.get("policy_port", client_profile.get("server_port", client_profile.get("port", 8000))))


# ============================================================================
# 配置值解析 / Configuration Value Resolution
# ============================================================================

def positive_int_or_none(value: Any, *, field_name: str) -> int | None:
    """将配置值解析为"正整数或 None"。

    用于需要"正数表示限制，null/None 表示不限制"的配置字段。
    例如最大步数：10 表示最多 10 步，None 表示无限。

    接受的 "null" 写法（大小写不敏感）：
      - Python None
      - 字符串 "none", "null", "unlimited"

    参数:
        value:      配置中读取的原始值
        field_name: 字段名称（用于生成清晰的错误信息）

    返回:
        int | None: 正整数或 None

    异常:
        ValueError: 如果值是零或负数

    示例:
        >>> positive_int_or_none(None, field_name="max_steps")
        None
        >>> positive_int_or_none("unlimited", field_name="max_steps")
        None
        >>> positive_int_or_none("5", field_name="max_steps")
        5
        >>> positive_int_or_none(-1, field_name="max_steps")
        ValueError: max_steps must be positive, null, or omitted
    """
    if value is None:
        return None
    # 如果传入的是字符串 null/none/unlimited，统一视为 None（无限制）
    if isinstance(value, str) and value.lower() in {"none", "null", "unlimited"}:
        return None
    steps = int(value)
    if steps <= 0:
        raise ValueError(f"{field_name} must be positive, null, or omitted")
    return steps


def task_name(config_profile: dict[str, Any]) -> str:
    """从配置中解析任务名称。

    解析优先级：
      1. client.task 字段（显式指定）
      2. 从 policy.config 字段中提取（去掉 "pi05_tron2_" 前缀）
      3. 原始 policy.config 值
      4. "tron2"（默认值）

    关于 "pi05_tron2_" 前缀：
      OpenPI 的策略配置名通常以 "pi05_tron2_" 开头（表示 π0.5 模型的 TRON2
      变体），去除前缀后剩下的就是纯任务名。
      例如 "pi05_tron2_pick_and_place" → "pick_and_place"

    参数:
        config_profile: 完整的配置字典

    返回:
        str: 任务名称

    示例:
        >>> task_name({"client": {"task": "grasping"}})
        "grasping"
        >>> task_name({"policy": {"config": "pi05_tron2_folding"}})
        "folding"
    """
    client_profile = section(config_profile, "client")
    policy_profile = section(config_profile, "policy")
    if client_profile.get("task"):
        return str(client_profile["task"])
    config = str(policy_profile.get("config") or "tron2")
    prefix = "pi05_tron2_"
    return config[len(prefix):] if config.startswith(prefix) else config


def record_paths(
    config_profile: dict[str, Any],
    *,
    action_key: str,
    state_key: str,
    action_suffix: str,
    state_suffix: str,
) -> tuple[Path, Path]:
    """生成带时间戳的录制文件路径（动作 CSV 和状态 CSV）。

    录制（recording）是 TRON2 数据收集流程的关键环节：
      - 动作文件（action CSV）：记录机器人的动作指令序列
      - 状态文件（state CSV）：记录对应的关节状态和时间戳

    文件命名规则：
      debug_images/{task_name}_{timestamp}_{suffix}.csv
      例如：debug_images/pick_and_place_20250805_143022_actions.csv

    调用此函数会自动创建文件所在的父目录（如果不存在）。
    文件路径也可以通过 client section 中的 action_key/state_key 字段显式指定。

    参数:
        config_profile: 完整的配置字典
        action_key:     client section 中指定动作文件路径的键名
        state_key:      client section 中指定状态文件路径的键名
        action_suffix:  动作文件的默认后缀（如 "actions"）
        state_suffix:   状态文件的默认后缀（如 "states"）

    返回:
        tuple[Path, Path]: (动作文件路径, 状态文件路径)

    示例:
        >>> record_paths(cfg, action_key="action_path", state_key="state_path",
        ...              action_suffix="actions", state_suffix="states")
        (Path("debug_images/pick_20250805_143022_actions.csv"),
         Path("debug_images/pick_20250805_143022_states.csv"))
    """
    client_profile = section(config_profile, "client")

    # 生成时间戳，精确到秒
    # 格式示例："20250805_143022"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    task = task_name(config_profile)

    # 构造默认文件路径
    default_action_path = Path("debug_images") / f"{task}_{timestamp}_{action_suffix}.csv"
    default_state_path = Path("debug_images") / f"{task}_{timestamp}_{state_suffix}.csv"

    # 如果配置中指定了路径就用配置的，否则用默认的
    action_path = Path(client_profile.get(action_key) or default_action_path)
    state_path = Path(client_profile.get(state_key) or default_state_path)

    # 确保父目录存在（mkdir -p 的效果）
    # parents=True  → 递归创建所有不存在的父目录
    # exist_ok=True → 如果目录已存在也不报错
    action_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    return action_path, state_path


# ============================================================================
# 观测数据格式化 / Observation Formatting
# ============================================================================

def format_obs(obs: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
    """将 TRON2 的观测数据格式化为 OpenPI WebSocket 推理服务器期望的格式。

    这个函数做了以下几件事：

      1. 浅拷贝顶层字段（跳过 "metadata" 元数据）
         - "metadata" 包含不需要发送给模型的额外信息，因此排除

      2. 浅拷贝 "images" 子字典中的每个图像数组
         - 保护原始数据不被后续操作修改

      3. 对每张图像进行处理：
         a. resize_with_pad(image, 224, 224)
            → 缩放到 224×224，用填充（padding）保持宽高比
         b. convert_to_uint8(...)
            → 确保像素值是 uint8 格式（0-255）
         c. einops.rearrange(img, "h w c -> c h w")
            → 把通道维从最后一维移到第一维
            → HWC（OpenCV/PIL 格式）→ CHW（PyTorch 格式）

      4. 可选的 prompt 注入
         - 如果提供了 prompt，将其添加到格式化后的字典中
         - prompt 会随观测一起发给推理服务器，影响模型的行为

    参数:
        obs:    TRON2 环境的原始观测字典
        prompt: 可选的文本指令（如 "pick up the red block"）

    返回:
        dict[str, Any]: 格式化后的观测字典，可直接发给推理服务器

    观测字典结构示意：
        obs = {
            "state": np.ndarray,       # 机器人关节状态
            "images": {
                "cam_high": np.ndarray,        # (H, W, C) → 变为 (C, H, W)
                "cam_left_wrist": np.ndarray,
                "cam_right_wrist": np.ndarray,
            },
            "metadata": {...},          # 会被过滤掉，不发送给模型
        }
    """
    # 步骤 1：拷贝顶层字段（排除 metadata）
    # np.ndarray.copy() 确保不修改原始数据
    formatted = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in obs.items()
        if k != "metadata"
    }

    # 步骤 2：拷贝 images 子字典
    formatted["images"] = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in obs.get("images", {}).items()
    }

    # 步骤 3：逐张图像 resize + 转换格式
    for cam_name, image in formatted["images"].items():
        # resize_with_pad：缩放并填充到 224x224，保持原始宽高比
        # convert_to_uint8：确保数据类型为 uint8（0-255 整数）
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
        # rearrange：HWC → CHW（OpenCV/PIL 格式 → PyTorch 格式）
        formatted["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

    # 步骤 4：注入 prompt（如果提供了的话）
    if prompt is not None:
        formatted["prompt"] = prompt

    return formatted


# ============================================================================
# 时间戳计算 / Timestamp Calculation Helpers
# ============================================================================

def timestamp_ms(value: Any) -> float:
    """安全地将任意值解析为毫秒时间戳。

    什么情况下时间戳可能无效？
      - None：没有接收到时间戳
      - 非数字字符串：解析失败
      - 零或负数：传感器还未初始化或时钟未同步

    这些情况统一返回 NaN（Not a Number），调用方可以通过 math.isnan()
    检查，避免因为无效时间戳导致崩溃。

    参数:
        value: 原始时间戳值（可能是 float、int、str 或 None）

    返回:
        float: 毫秒时间戳，或 NaN（表示无效）

    示例:
        >>> timestamp_ms(1700000000000.0)
        1700000000000.0
        >>> timestamp_ms(None)
        nan
        >>> timestamp_ms("invalid")
        nan
        >>> timestamp_ms(0)
        nan
    """
    if value is None:
        return float("nan")
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return float("nan")
    # 时间戳必须为正数（0 和负数表示无效/未初始化）
    return timestamp if timestamp > 0 else float("nan")


def age_ms(receive_wall_s: float, timestamp_ms_value: float) -> float:
    """计算观测数据的"年龄"（从产生到接收经历了多少毫秒）。

    公式：
        age_ms = 接收时的墙上时间 − 数据中的时间戳

    如果数据时间戳无效（NaN），返回 NaN。

    这个指标用于衡量网络延迟和数据新鲜度：
      - 小值（<10ms） → 数据很新鲜，延迟低
      - 大值（>100ms） → 数据较旧，可能有网络延迟或处理瓶颈

    参数:
        receive_wall_s:   接收该观测时的墙上时间（秒，time.time()）
        timestamp_ms_value: 数据中携带的时间戳（毫秒）

    返回:
        float: 数据年龄（毫秒），或 NaN

    示例:
        >>> age_ms(1700000000.5, 1700000000000.0)  # 刚产生 500ms
        500.0
        >>> age_ms(1700000000.5, float("nan"))
        nan
    """
    if math.isnan(timestamp_ms_value):
        return float("nan")
    return receive_wall_s * 1000.0 - timestamp_ms_value


def relative_sensor_time_s(receive_rel_s: float, age_ms_value: float) -> float:
    """计算传感器的相对时间（相对于某个参考点的秒数）。

    公式：
        sensor_time = 接收相对时间 − 数据年龄（转换为秒）

    用途：
      在离线回放或数据分析中，需要知道数据"实际来自什么时间"。
      相对时间比绝对时间戳更适合用于排序和对齐。

    参数:
        receive_rel_s: 接收时相对于某个起点的秒数（如 time.monotonic()）
        age_ms_value:   数据的年龄（毫秒）

    返回:
        float: 传感器相对时间（秒），或 NaN

    示例:
        >>> relative_sensor_time_s(100.0, 500.0)  # 100s 时收到 500ms 前产生的数据
        99.5
        >>> relative_sensor_time_s(100.0, float("nan"))
        nan
    """
    if math.isnan(age_ms_value):
        return float("nan")
    return receive_rel_s - age_ms_value / 1000.0


# ============================================================================
# Prompt 清理 / Prompt Sanitization
# ============================================================================

def _clean_prompt(prompt: str | None) -> str | None:
    """清理 prompt 字符串：去首尾空白，空串视为 None。

    内部辅助函数。确保 prompt 要么是一个有意义非空字符串，要么是 None。
    避免空字符串被当作有效 prompt 发送给推理服务器。

    参数:
        prompt: 原始 prompt 字符串或 None

    返回:
        str | None: 清理后的 prompt，或 None（如果原始值无效）

    示例:
        >>> _clean_prompt("  pick up the block  ")
        "pick up the block"
        >>> _clean_prompt("   ")
        None
        >>> _clean_prompt(None)
        None
    """
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


# ============================================================================
# Prompt 控制器 / Prompt Controller
# ============================================================================

class PromptController:
    """线程安全的 prompt 管理器，支持运行时通过 stdin 实时切换任务指令。

    使用场景：
      TRON2 机器人在执行任务时，操作者可能想在不重启程序的情况下切换
      任务指令。例如：
        - 当前在 "pick up the red block"
        - 操作者直接在终端输入 "place it on the table"
        - 下一个推理周期就会使用新的 prompt

    线程安全设计：
      - get() 和 set() 都用 threading.Lock 保护
      - 后台线程（daemon）读取 stdin，不会阻塞主线程
      - 后台线程通过 set() 更新 prompt，线程安全

    如果没有可用的终端（非 TTY），后台监听会自动禁用。

    使用示例：
        controller = PromptController("pick up the block")
        controller.start_stdin_listener()

        while True:
            prompt = controller.get()  # 获取当前 prompt（可能已被用户更新）
            action = model.infer(obs, prompt=prompt)
            ...

    属性:
        _lock:   threading.Lock — 保护 _prompt 的读写
        _prompt: str | None — 当前的 prompt 文本
        _thread: threading.Thread | None — stdin 监听线程（只启动一次）
    """

    def __init__(self, initial: str | None = None):
        """初始化 PromptController。

        参数:
            initial: 初始 prompt（可选）。空字符串或纯空白会被视为 None。
        """
        self._lock = threading.Lock()         # 互斥锁：保护 _prompt 字段
        self._prompt = _clean_prompt(initial)  # 清理后的初始 prompt
        self._thread: threading.Thread | None = None  # stdin 监听线程（懒初始化）

    def get(self) -> str | None:
        """线程安全地获取当前 prompt。

        返回:
            str | None: 当前 prompt，或 None（表示没有有效的 prompt）
        """
        with self._lock:
            return self._prompt

    def set(self, prompt: str | None) -> None:
        """线程安全地设置新的 prompt。

        空字符串或纯空白会被自动清理为 None。

        参数:
            prompt: 新的 prompt 文本
        """
        with self._lock:
            self._prompt = _clean_prompt(prompt)

    def start_stdin_listener(self) -> None:
        """启动后台线程，监听 stdin 以获取实时 prompt 更新。

        行为：
          - 如果后台线程已启动：直接返回（幂等操作）
          - 如果 sys.stdin 不可用或不是终端（TTY）：
            记录日志信息，禁用实时输入功能
          - 否则：启动一个 daemon 线程，阻塞读取 stdin
            - daemon=True 意味着主程序退出时该线程自动终止
            - 每读取一行非空文本，就更新 prompt

        注意：
          - 这个方法最多只能启动一次监听线程
          - stdin 读取是阻塞的，所以必须放在独立线程中
            （如果放在主线程，会阻塞机器人的控制循环）
        """
        # 已经启动了就不重复启动（幂等性）
        if self._thread is not None:
            return

        # 检查是否有可用的终端
        # sys.stdin.isatty()：判断 stdin 是否连接到终端（交互式环境）
        # 如果 stdin 被重定向（如从文件读取），则无法交互输入
        if not sys.stdin or not sys.stdin.isatty():
            logger.info("Live prompt input disabled; using prompt=%r.", self._prompt)
            return

        # 启动 daemon 线程读取 stdin
        # daemon=True：主线程结束时自动退出，不用担心僵尸线程
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="PromptInput")
        self._thread.start()
        logger.info("Live prompt input enabled. Type a new prompt then Enter to switch.")

    def _reader_loop(self) -> None:
        """后台线程的主循环：逐行读取 stdin 并更新 prompt。

        这是运行在独立线程中的方法。它会阻塞在 sys.stdin 的迭代器上，
        直到 stdin 关闭（EOF）或线程被终止。

        行为：
          - 忽略空行（直接跳过，不清除当前 prompt）
          - 非空行会被 strip 后设为新 prompt
          - 如果读取过程中发生异常，记录错误并退出循环
        """
        try:
            # sys.stdin 是可迭代的，每次迭代读取一行
            # 这个迭代会阻塞等待用户输入
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    # 空行忽略（方便用户按 Enter 跳过）
                    continue
                self.set(text)
                # 记录切换后的 prompt，方便用户确认
                logger.info("[PROMPT] Updated task prompt -> %r", self.get())
        except Exception:
            # 例如 stdin 被关闭、线程被强制终止等
            logger.exception("Prompt stdin listener stopped.")


# ============================================================================
# 推理计时 / Inference Timing
# ============================================================================

def infer_with_timing(policy, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """执行一次 WebSocket 推理并测量客户端侧的传输计时。

    这个函数调用推理策略的底层 WebSocket 接口，并在每个步骤记录时间戳，
    用于诊断推理延迟的瓶颈所在。

    测量的步骤（按顺序）：
      ┌──────────────┬──────────────────────────────────┐
      │ 指标            │ 含义                                │
      ├──────────────┼──────────────────────────────────┤
      │ pack_ms         │ 用 msgpack 序列化观测数据的时间       │
      │ send_ms         │ WebSocket 发送数据的时间             │
      │ recv_wait_ms    │ 等待服务器响应的时间（包含推理时间）   │
      │ unpack_ms       │ 用 msgpack 反序列化响应的时间         │
      │ total_ms        │ 总耗时（pack + send + wait + unpack）│
      │ payload_kb      │ 发送的数据大小（KB）                  │
      │ response_kb     │ 接收的数据大小（KB）                  │
      └──────────────┴──────────────────────────────────┘

    延迟分析指南：
      - recv_wait_ms 最大 → 瓶颈在服务器端（模型推理慢）
      - pack_ms 或 unpack_ms 大 → 瓶颈在序列化（图像太多或太大）
      - send_ms 大 → 网络带宽不足
      - payload_kb 大 → 考虑压缩图像或降低分辨率

    参数:
        policy: 推理策略对象（需要有 _packer 和 _ws 属性）
        obs:    格式化后的观测字典（通常由 format_obs() 产生）

    返回:
        tuple[dict[str, Any], dict[str, float]]:
          (推理结果字典, 计时指标字典)

    异常:
        RuntimeError: 如果服务器返回了错误字符串而非序列化的响应
    """
    # 延迟导入：只在需要时才导入 msgpack_numpy
    # 避免在仅使用配置功能的场景中依赖 msgpack
    from openpi_client import msgpack_numpy

    # t0: 开始打包（序列化观测数据）
    t0 = time.perf_counter()
    # _packer.pack() 将 Python 字典（包含 numpy 数组）序列化为 msgpack 字节
    # msgpack_numpy 支持高效的 numpy 数组序列化
    data = policy._packer.pack(obs)
    t1 = time.perf_counter()

    # 通过 WebSocket 发送序列化后的数据
    policy._ws.send(data)
    t2 = time.perf_counter()

    # 接收 WebSocket 响应（阻塞等待服务器返回）
    # 这段时间包含了：网络传输 + 模型推理 + 结果序列化
    response = policy._ws.recv()
    t3 = time.perf_counter()

    # 检查是否为错误响应（字符串类型表示错误）
    if isinstance(response, str):
        raise RuntimeError(f"Error in inference server:\n{response}")

    # 反序列化响应
    ans = msgpack_numpy.unpackb(response)
    t4 = time.perf_counter()

    # 返回推理结果和计时指标
    return ans, {
        "pack_ms": (t1 - t0) * 1000.0,            # 打包耗时（ms）
        "send_ms": (t2 - t1) * 1000.0,            # 发送耗时（ms）
        "recv_wait_ms": (t3 - t2) * 1000.0,       # 等待响应耗时（ms）—— 含模型推理
        "unpack_ms": (t4 - t3) * 1000.0,          # 解包耗时（ms）
        "total_ms": (t4 - t0) * 1000.0,           # 总耗时（ms）
        "payload_kb": len(data) / 1024.0,          # 发送数据量（KB）
        "response_kb": len(response) / 1024.0,     # 响应数据量（KB）
    }
