"""OpenPI 部署 YAML 配置文件的加载与解析工具。

=============================================================================
背景 / Background
=============================================================================
本模块是 OpenPI 部署配置系统的"底层"——它直接读写 YAML 文件，提供
路径解析、配置加载、section 提取和布尔值解析等基础功能。

上层模块（如 examples/tron2/deploy_config.py）调用本模块的函数，
再进一步构建强类型的配置对象（Tron2Config、CameraConfig 等）。

模块功能概览：
  1. REPO_ROOT          — 定位 OpenPI 项目根目录的路径常量
  2. resolve_config_path — 将用户指定的配置文件路径解析为绝对路径
  3. load_deploy_config  — 加载 YAML 文件并返回 Python 字典
  4. section             — 从配置字典中安全地提取子配置块
  5. bool_value          — 将 YAML 中的多种布尔写法统一转换为 Python bool

为什么需要这个模块？
  部署配置文件（YAML）是 OpenPI 系统的"单一事实来源"（single source of
  truth）。它包含了机器人 IP、相机配置、Bridge 地址等所有必要参数。
  本模块确保无论用户从哪里运行脚本、用相对路径还是绝对路径，配置文件
  都能被正确找到和解析。
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# PyYAML：Python 的 YAML 解析库
# safe_load() 只解析基础的 YAML 类型（dict、list、str、int、float、bool、null），
# 不会执行任意的 Python 对象构造，因此是安全的。
import yaml


# ============================================================================
# 项目根目录定位
# ============================================================================

# REPO_ROOT：OpenPI 项目的根目录（绝对路径）
#
# 计算方式：
#   __file__            → src/openpi/shared/deploy_config.py（可能是相对路径）
#   .resolve()          → 转为绝对路径
#   .parents[3]         → 向上走 3 层父目录
#
# 以本项目实际结构为例：
#   __file__    = .../openpi/src/openpi/shared/deploy_config.py
#   parents[0]  = .../openpi/src/openpi/shared/    （当前文件所在目录）
#   parents[1]  = .../openpi/src/openpi/            （openpi 包目录）
#   parents[2]  = .../openpi/src/                   （源码根目录）
#   parents[3]  = .../openpi/                       ← REPO_ROOT
#
# 路径层级示意图：
#
#   REPO_ROOT/                                     ← parents[3]
#   ├── src/
#   │   └── openpi/
#   │       └── shared/
#   │           └── deploy_config.py  ← __file__ (parents[0])
#   ├── examples/
#   ├── configs/           ← 默认的 YAML 配置文件存放处
#   ├── pyproject.toml
#   └── ...
#
# 为什么要定位 REPO_ROOT？
#   用户可能从任何目录运行脚本，如果使用相对路径指定配置文件，
#   我们需要一个"锚点"来解析路径。REPO_ROOT 就是这个锚点——
#   相对于项目根目录查找配置文件。
REPO_ROOT = Path(__file__).resolve().parents[3]


# ============================================================================
# 配置文件路径解析 / Configuration Path Resolution
# ============================================================================

def resolve_config_path(path: str | Path) -> Path:
    """将用户指定的配置文件路径解析为确定的绝对路径。

    解析策略（按优先级尝试）：

      1. 如果传入的是绝对路径（以 / 开头）：
         → 展开 ~（用户主目录）后直接返回
         例如：/home/user/my_config.yaml → /home/user/my_config.yaml

      2. 如果当前工作目录下存在该文件：
         → 返回 {当前工作目录}/{传入路径}
         例如：用户从 /home/user/project 运行，传入 "config.yaml"
              → /home/user/project/config.yaml

      3. 否则：
         → 返回 {REPO_ROOT}/{传入路径}
         例如：传入 "configs/tron2.yaml"
              → /path/to/openpi/configs/tron2.yaml

    优先级设计的原因：
      - 当前工作目录优先：如果用户在自己创建的目录下运行，并放置了
        自定义配置文件，应该优先使用它（显式 > 隐式）
      - REPO_ROOT 兜底：如果当前目录没有，就从项目根目录找——
        这通常是默认配置文件的位置

    参数:
        path: 用户指定的路径（字符串或 Path 对象），可以是：
              - 绝对路径："/home/user/config.yaml"
              - 相对路径："configs/tron2.yaml"
              - 带 ~ 的路径："~/my_configs/robot.yaml"

    返回:
        Path: 解析后的绝对路径

    示例:
        >>> resolve_config_path("/absolute/path/config.yaml")
        PosixPath('/absolute/path/config.yaml')

        >>> resolve_config_path("~/my_config.yaml")
        PosixPath('/home/user/my_config.yaml')

        >>> # 如果当前目录有 config.yaml 就用它，否则找 REPO_ROOT/config.yaml
        >>> resolve_config_path("config.yaml")
        PosixPath('/current/working/dir/config.yaml')  # 或 REPO_ROOT/config.yaml
    """
    # 展开用户主目录符号 ~
    # 例如 "~/config.yaml" → "/home/user/config.yaml"
    profile_path = Path(path).expanduser()

    # 如果是绝对路径，直接返回（无需再做任何解析）
    if profile_path.is_absolute():
        return profile_path

    # 优先级 1：尝试在当前工作目录下查找
    # Path.cwd() 返回运行脚本时的工作目录（不是脚本所在目录）
    cwd_path = Path.cwd() / profile_path
    if cwd_path.exists():
        return cwd_path

    # 优先级 2：从项目根目录查找（兜底策略）
    return REPO_ROOT / profile_path


# ============================================================================
# YAML 配置加载 / YAML Configuration Loading
# ============================================================================
def select_profile_path(
    profile: str | Path | None,
    deploy_config: str | Path | None = None,
) -> str | Path | None:
    """Return the preferred deploy profile path while preserving the old flag."""
    if profile is not None and deploy_config is not None:
        raise ValueError("Use either --profile or --deploy-config, not both.")
    return profile if profile is not None else deploy_config


def load_deploy_config(path: str | Path | None) -> dict[str, Any]:
    """加载部署 YAML 配置文件，返回 Python 字典。

    如果 path 为 None，直接返回空字典 {} —— 这允许调用方在不需要
    自定义配置时省略配置文件，所有参数都使用默认值。

    安全性：
      使用 yaml.safe_load() 而非 yaml.load()：
        - safe_load 只解析 YAML 基础类型（dict、list、str、int、float、
          bool、null），不解析 YAML 标签（!!python/object 等）
        - 这可以防止 YAML 反序列化攻击（任意代码执行）
        - 对于配置文件场景完全够用

    参数:
        path: YAML 文件路径（可以是相对路径、绝对路径或 None）

    返回:
        dict[str, Any]: 解析后的配置字典。如果 path 为 None，返回 {}。

    异常:
        FileNotFoundError: 文件不存在（由 resolve_config_path 或 open 触发）
        ValueError: YAML 文件的顶层结构不是字典（mapping）

    示例:
        >>> config = load_deploy_config("configs/tron2.yaml")
        >>> print(config.keys())
        dict_keys(['robot', 'camera', 'bridge', 'client', 'policy'])

        >>> config = load_deploy_config(None)  # 无配置文件，全部用默认值
        >>> config
        {}
    """
    if path is None:
        return {}

    # 解析路径 → 绝对路径
    resolved_path = resolve_config_path(path)

    # 打开并解析 YAML 文件
    with resolved_path.open() as f:
        # safe_load：安全的 YAML 解析（不执行任意代码）
        # or {}：如果 YAML 文件是空的（safe_load 返回 None），用空字典代替
        data = yaml.safe_load(f) or {}

    # 校验：YAML 文件的顶层必须是键值对（mapping/dict）
    # 不允许顶层是列表或纯量值，因为我们期望的是分 section 的配置结构
    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {resolved_path}")
    return data


# ============================================================================
# 配置 Section 提取 / Configuration Section Extraction
# ============================================================================

def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """从完整配置字典中安全地提取指定名称的子配置块（section）。

    配置文件的结构通常是嵌套的，例如：

        robot:
          ip: "192.168.1.100"
          port: 5000
        camera:
          resolution: [480, 640, 3]
        client:
          fps: 30.0

    调用 section(config, "robot") 返回 {"ip": "192.168.1.100", "port": 5000}。

    安全处理：
      1. section 不存在 → 返回空字典 {}（而非报错）
      2. section 的值为 None → 返回空字典 {}
      3. section 的值不是字典 → 抛出 ValueError（类型不匹配）

    为什么不存在时返回 {} 而不是报错？
      这样调用方可以安全地使用 .get() 链式获取嵌套值：
        section(config, "camera").get("resolution", [480, 640, 3])
      如果 camera section 不存在，section() 返回 {}，
      .get("resolution", [480, 640, 3]) 就会返回默认值 [480, 640, 3]。
      全程不需要 try/except，代码更简洁。

    参数:
        config: 完整的配置字典
        name:   section 的名称（如 "robot"、"camera"、"bridge"、"client"）

    返回:
        dict[str, Any]: 提取出的子配置字典。如果不存在，返回 {}。

    异常:
        ValueError: 如果 section 存在但不是字典类型

    示例:
        >>> config = {"robot": {"ip": "192.168.1.100"}, "version": 2}
        >>> section(config, "robot")
        {'ip': '192.168.1.100'}

        >>> section(config, "missing_section")
        {}

        >>> section(config, "version")
        ValueError: Deploy config section must be a mapping: version
    """
    value = config.get(name, {})

    # YAML 中 section 的值可能显式写为 null/None
    # 例如：camera: null  或  camera: ~
    # 此时也返回空字典，与 section 不存在的行为保持一致
    if value is None:
        return {}

    # 类型校验：section 的值必须是字典
    # 防止用户错误地把列表或纯量写成 section
    # 例如 camera: [480, 640] 是不合法的配置写法
    if not isinstance(value, dict):
        raise ValueError(f"Deploy config section must be a mapping: {name}")
    return value


# ============================================================================
# 布尔值解析 / Boolean Value Resolution
# ============================================================================

def bool_value(value: Any) -> bool:
    """将 YAML 中的多种"布尔"写法统一解析为 Python bool。

    问题背景：
      YAML 和不同用户习惯中有多种表示布尔值的方式：
        - YAML 原生：true, false, yes, no, on, off
        - 环境变量风格："1", "0", "TRUE", "FALSE"
        - 用户手写："y", "n", "yes", "no"
        - 整数：1, 0

    如果直接用 Python 的 bool() 转换：
      bool("false") → True   （非空字符串都是 True！）
      bool("0")     → True   （同上）
      bool(0)       → False  （整数 0 是 False）

    这会导致配置中的 "false"（用户显然希望是 False）被错误地解释为 True。
    本函数专门处理这些情况，确保用户意图得到正确理解。

    解析规则（按优先级）：
      1. 如果已经是 Python bool → 直接返回
      2. 如果是字符串且值为 "1", "true", "yes", "y", "on"（大小写不敏感）
         → 返回 True
      3. 其他字符串（包括 "false", "no", "off", "0" 等） → 返回 False
      4. 其他类型 → 使用 Python 原生的 bool() 转换

    第 4 条主要处理整数：0 → False，非 0 → True。

    参数:
        value: 从 YAML 配置中读取的任意值

    返回:
        bool: 标准 Python 布尔值

    示例:
        >>> bool_value(True)
        True
        >>> bool_value("yes")
        True
        >>> bool_value("false")
        False
        >>> bool_value("0")
        False
        >>> bool_value(1)
        True
        >>> bool_value(0)
        False
        >>> bool_value("ON")
        True
    """
    # 已经是 Python bool → 直接返回（最常见的情况：YAML 解析后就是 bool）
    if isinstance(value, bool):
        return value

    # 字符串类型：检查是不是"肯定的"布尔值
    # 只有明确列出的这几个值（大小写不敏感）才算 True
    # 其余所有字符串都算 False（包括 "false", "no", "off", "0" 等）
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}

    # 其他类型（通常是 int/float）：使用 Python 原生的 bool 转换
    # bool(0) → False, bool(1) → True, bool(2) → True
    return bool(value)
