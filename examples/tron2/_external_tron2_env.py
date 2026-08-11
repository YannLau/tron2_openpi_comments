"""Make the sibling tron2_env package importable from TRON2 examples.

=============================================================================
背景 / Background
=============================================================================
本项目的目录结构大致如下（workspace 级别）：

    workspace/                        ← workspace_root
    ├── openpi/                       ← openpi_root（本项目）
    │   ├── examples/
    │   │   └── tron2/
    │   │       ├── _external_tron2_env.py   ← 你正在看的这个文件
    │   │       └── ...（其他 TRON2 示例）
    │   └── ...
    └── tron2_env/                   ← 兄弟项目 / sibling project
        └── src/                      ← tron2_env 包的源码所在
            └── tron2_env/
                └── __init__.py

TRON2 示例代码需要 `import tron2_env`，但 `tron2_env` 并不在本项目
（openpi）内部，而是位于同级的兄弟目录 `tron2_env/` 中。

默认情况下，Python 只会从以下位置搜索模块：
  1. 当前脚本所在目录
  2. `PYTHONPATH` 环境变量中的路径
  3. 标准库路径
  4. site-packages（pip 安装的第三方包）

兄弟项目 `tron2_env/src` 不在上述任何位置，因此直接 import 会报
ModuleNotFoundError。

本文件的作用就是：在示例代码运行之前，将 `tron2_env/src` 临时加入
`sys.path`，让 Python 能够找到并导入 `tron2_env` 包。
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_external_tron2_env_on_path() -> None:
    """将兄弟项目 tron2_env 的源码目录添加到 sys.path，使其可被 import。

    这个函数是幂等的（idempotent）——多次调用不会重复添加相同的路径。
    它只在 tron2_env 目录确实存在时才添加，因此即使 tron2_env 尚未克隆，
    调用本函数也不会报错（只是无法 import tron2_env，届时会有清晰的报错）。
    """

    # ------------------------------------------------------------------
    # 第 1 步：定位 openpi 项目的根目录
    # ------------------------------------------------------------------
    # __file__            → 当前文件的路径（可能是相对路径）
    # .resolve()          → 转为绝对路径
    # .parents[2]         → 向上走 2 层父目录
    #
    # 以本项目实际结构为例：
    #   __file__  = .../workspace/openpi/examples/tron2/_external_tron2_env.py
    #   parents[0] = .../workspace/openpi/examples/tron2/
    #   parents[1] = .../workspace/openpi/examples/
    #   parents[2] = .../workspace/openpi/                ← openpi_root
    # ------------------------------------------------------------------
    openpi_root = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # 第 2 步：定位 workspace 根目录
    # ------------------------------------------------------------------
    # openpi_root.parent = .../workspace/                 ← workspace_root
    #
    # 之所以要通过 openpi_root 的 parent 来定位 workspace，而不是用
    # parents[3]，是因为：
    #   - 代码意图更清晰：workspace = openpi 的父目录
    #   - 如果未来文件层级变动（比如文件挪到 examples/ 直接下面），
    #     只需要改一处 parents 索引即可
    # ------------------------------------------------------------------
    workspace_root = openpi_root.parent

    # ------------------------------------------------------------------
    # 第 3 步：确定候选路径（可能包含 tron2_env 包的位置）
    # ------------------------------------------------------------------
    # 首选：workspace/tron2_env/src/
    #   - 这是 tron2_env 的 Python 源码目录，里面有 tron2_env/__init__.py
    #   - 将此路径加入 sys.path 后，`import tron2_env` 即可正常工作
    #
    # 备选：workspace/
    #   - 在某些部署场景下，tron2_env 可能以不同方式组织
    #   - 将 workspace 根目录加入 sys.path 作为兜底
    # ------------------------------------------------------------------
    candidate_paths = [
        workspace_root / "tron2_env" / "src",
        workspace_root,
    ]

    # ------------------------------------------------------------------
    # 第 4 步：将存在的路径插入 sys.path 最前面
    # ------------------------------------------------------------------
    # `reversed(candidate_paths)` 的作用：
    #   候选列表的顺序是 [首选, 备选]，但 sys.path 的搜索顺序是从前到后的。
    #   如果正向遍历，首选会先被插入到 sys.path[0]，然后备选被插入到
    #   sys.path[0]（更前面），导致备选反而排在首选之前。
    #   反向遍历后，备选先插入到 sys.path[0]，首选再插入到 sys.path[0]，
    #   最终首选排在备选前面 → Python 优先搜索首选路径。✓
    #
    # sys.path.insert(0, path_str) vs sys.path.append(path_str)：
    #   insert(0, ...) 将路径插入到 sys.path 的最前面，这意味着该路径的
    #   优先级最高——如果其他地方也有名为 tron2_env 的包，我们插入的这
    #   个会被优先使用。这正是我们想要的行为。
    #
    # 检查条件：
    #   1. path.exists()          → 路径确实存在于磁盘上（否则跳过）
    #   2. path_str not in sys.path → 尚未添加过（保证幂等性）
    # ------------------------------------------------------------------
    for path in reversed(candidate_paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
