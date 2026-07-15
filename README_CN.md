# TRON2 OpenPI 部署仓库

[English](README.md)

`tron2_openpi` 是基于 OpenPI 改进的 TRON2 部署仓库。它保留 OpenPI 的
policy serving、pi0/pi0.5 模型栈和客户端基础能力，并加入 TRON2 policy
transform、部署配置模板和 TRON2 真机客户端示例。

本仓库需要和同级的 `tron2_env` 运行时包一起使用。它的定位是集成和部署示例，
不是私有 checkpoint、数据集、low-level 机器人 SDK 或本地部署配置的完整发布。

## 本仓库包含什么

- 通过 `scripts/serve_policy.py` 启动 pi0.5 策略服务。
- `src/openpi/policies/tron2_policy.py` 中的 TRON2 policy 输入/输出转换。
- `src/openpi/training/config.py` 中的 TRON2 训练/部署配置注册。
- `examples/tron2/` 中的 TRON2 真机客户端。
- `configs/deploy/` 中的公开部署配置模板。
- 通过 `scripts/train_tron2_task.py` 和 `configs/train/tron2_tasks/example.yaml`
  使用 YAML 配置新的 TRON2 训练任务。
- 可选的 Bridge 观测模式：从 TRON2 Bridge 获取图像和状态。
- 可选的 legacy RealSense 观测模式：使用本机直连相机。
- RTC 部署客户端，包含 warmup、观测超时恢复、队列诊断和可选动作平滑。
- `packages/openpi-client/` 中的 OpenPI client 包。

## 本仓库不包含什么

- 模型权重和 checkpoint 目录。
- 训练数据集、评测数据集、日志或 benchmark 结果。
- 私有 `.local.yaml` 部署文件。
- 凭据、真实相机序列号、客户数据或私有本地部署配置。
- 尚未开发完成的 low-level 机器人 transport。
- 无人值守真机运行的安全认证。

## 目录结构

```text
同级目录/
├── tron2_openpi/
│   ├── configs/
│   │   ├── deploy/
│   │   │   ├── candy.yaml
│   │   │   ├── tron2_deploy.example.yaml
│   │   │   └── tron2_deploy.example_CN.yaml
│   │   └── train/
│   │       └── tron2_tasks/
│   │           └── example.yaml
│   ├── examples/
│   │   └── tron2/
│   │       ├── deploy_config.py
│   │       ├── pi_client.py
│   │       └── pi_client_rtc.py
│   ├── packages/
│   │   └── openpi-client/
│   ├── scripts/
│   │   ├── cloud_train_entrypoint_portable.sh
│   │   ├── compute_norm_stats.py
│   │   ├── serve_policy.py
│   │   └── train_tron2_task.py
│   └── src/
│       └── openpi/
└── tron2_env/
```

请保持 `tron2_openpi/` 和 `tron2_env/` 两个目录同级。TRON2 客户端启动时会把
同级 `../tron2_env/src` 加入 `sys.path`，因此可以直接导入运行时包。
录制动作回放工具位于 `../tron2_env/examples/replay_data.py`。

## 环境要求

- 主要部署环境为 Ubuntu 22.04。
- Python 3.11 或更新版本。
- 使用 `uv` 管理 Python 依赖。
- 策略推理需要 NVIDIA GPU 和兼容 CUDA 的 JAX 环境。
- 客户端机器需要能访问 TRON2 WebSocket 机器人控制器。
- 使用 `client.observation_source: bridge` 时需要访问 TRON2 Bridge。
- 使用 `client.observation_source: legacy` 时需要 Intel RealSense 相机和本机相机访问权限。

如果系统中还没有 `uv`，可以安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 环境安装

将本仓库和独立的 TRON2 运行时仓库 clone 到同级目录，然后安装环境：

```bash
git clone https://github.com/limx-tron2/tron2_openpi.git
git clone https://github.com/limx-tron2/tron2_env.git

cd tron2_openpi
uv sync
uv pip install -e .

cd ../tron2_env
python -m pip install -e ".[bridge,openpi]"
```

验证同级运行时能被导入：

```bash
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run python -c "import tron2_env; print(tron2_env.__file__)"
```

输出路径应该指向 `../tron2_env/src/tron2_env/__init__.py`。

## 部署配置

请从公开模板开始：

- Candy 任务公开示例：`configs/deploy/candy.yaml`
- 英文模板：`configs/deploy/tron2_deploy.example.yaml`
- 中文模板：`configs/deploy/tron2_deploy.example_CN.yaml`

`configs/deploy/candy.yaml` 是 Candy 任务的公开示例配置，可以包含机器人局域网 IP 和任务相关示例值。用于你自己的机器人前，请确认 checkpoint 路径、机器人地址、Bridge 地址、初始化姿态和安全流程都适用于当前部署。

复制一份本地私有配置，并只修改 local 文件：

```bash
cp configs/deploy/tron2_deploy.example_CN.yaml configs/deploy/tron2_deploy.local.yaml
```

不要提交 `.local.yaml` 文件。这类文件用于保存私有路径、机器人地址、Bridge 地址和相机序列号。

最小本地配置示例：

```yaml
policy:
  config: TASK_CONFIG_NAME
  repo_id: DATASET_REPO_ID
  checkpoint_dir: /path/to/checkpoints/TASK_CONFIG_NAME/experiment/step
  default_prompt: TASK_PROMPT

robot:
  ip: ROBOT_IP

client:
  observation_source: bridge
  max_steps: null
```

重要字段：

| 字段 | 说明 |
| --- | --- |
| `policy.config` | `src/openpi/training/config.py` 中注册的训练配置名。 |
| `policy.repo_id` | 用于加载归一化统计的 assets 目录名。 |
| `policy.checkpoint_dir` | 训练好的 checkpoint step 目录。 |
| `policy.default_prompt` | 客户端未传 `--prompt` 时使用的默认语言指令。 |
| `policy.record` | 为 `true` 时保存原始 policy 输入/输出，用于调试。 |
| `policy.action_horizon` | 可选的推理 action chunk 长度覆盖项。 |
| `policy.state_dim` | 可选的 TRON2 state/action 输出维度覆盖项。 |
| `policy.use_delta_joint_actions` | 可选的 delta action transform 覆盖项。 |
| `server.host` / `server.port` | policy server 监听地址。 |
| `client.policy_host` / `client.policy_port` | 客户端看到的 policy server 地址。 |
| `client.observation_source` | `bridge` 或 `legacy`。 |
| `client.state_dim` | `16` 表示双臂和夹爪，`18` 表示额外包含头部关节。 |
| `client.fps` | policy action 播放频率。 |
| `client.publish_rate` | 后台 ServoJ 指令发送频率。 |
| `client.max_steps` | 运行多少个 policy chunk；`null` 表示持续运行直到手动停止。 |
| `client.rtc_enabled` | 为 `true` 时使用 `pi_client_rtc.py`；为 `false` 时使用 `pi_client.py`。 |
| `client.duration` | RTC 运行时长，单位秒；`0` 表示一直运行。 |
| `client.execution_horizon` / `client.delay` | RTC 的 `s` 和初始 `d` 时序参数。 |
| `client.rtc_guidance_enabled` | 是否启用推理时 RTC VJP guidance。 |
| `client.trained_rtc_mode` | checkpoint 使用训练时 RTC 时打开该模式。 |
| `robot.ip` / `robot.port` | TRON2 WebSocket 机器人控制器地址。 |
| `bridge.host` | Bridge 观测模式使用的 TRON2 Bridge WebSocket 地址。 |
| `camera.serial_to_name` | legacy 模式下 RealSense 序列号到 policy 相机名的映射。 |

`policy.repo_id` 必须和 checkpoint 内的 assets 目录一致：

```text
checkpoint_dir/assets/<policy.repo_id>/norm_stats.json
```

当前代码中注册的 TRON2 示例配置：

| `policy.config` | `policy.repo_id` |
| --- | --- |
| `pi05_tron2_alarm` | `alarm` |
| `pi05_tron2_Banana` | `banana` |
| `pi05_tron2_cabinet` | `cabinet` |
| `pi05_tron2_Candy` | `candy` |
| `pi05_tron2_Chess` | `chess` |
| `pi05_tron2_Cloth` | `cloth` |
| `pi05_tron2_Drawer` | `drawer` |
| `pi05_tron2_Duck` | `duck` |
| `pi05_tron2_SortFruit` | `sort` |

## 观测模式

Bridge 模式从 TRON2 Bridge 获取图像，默认也使用 Bridge 对齐后的状态：

```yaml
client:
  observation_source: bridge

bridge:
  host: wss://BRIDGE_HOST
  state_source: bridge
```

Legacy 模式使用本机直连 RealSense 相机，并从机器人 WebSocket 获取状态：

```yaml
client:
  observation_source: legacy

camera:
  serial_to_name:
    HEAD_CAMERA_SERIAL: cam_high
    LEFT_WRIST_CAMERA_SERIAL: cam_left_wrist
    RIGHT_WRIST_CAMERA_SERIAL: cam_right_wrist
```

TRON2 policy 期望的图像 key 为：

- `cam_high`
- `cam_left_wrist`
- `cam_right_wrist`

## 启动策略服务

启动 policy server：

```bash
uv run scripts/serve_policy.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

在另一个终端启动 TRON2 客户端：

**注意：机器人需处于L1+X后的初始状态，然后切换到高级开发者模式，运行客户端后机器人会侧展双臂，然后前伸，若非初始状态可能会直接抬起双臂，警惕前方物体风险！！！**

```bash
uv run python examples/tron2/pi_client.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

临时覆盖任务指令：

```bash
uv run python examples/tron2/pi_client.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml \
  --prompt="put the object into the drawer"
```

当 `client.max_steps` 为 `null` 时，需要手动停止客户端。

## RTC 部署

RTC 使用同一个 server 命令。server 会自动检测加载的模型是否支持 RTC，并在
websocket metadata 中发布 `rtc_enabled` 和 `action_horizon`。client 侧运行参数来自
YAML。

在本地 YAML 中设置：

```yaml
client:
  rtc_enabled: true
  duration: 120
  fps: 30
  execution_horizon: 10
  delay: 2
  rtc_guidance_enabled: true
  rtc_guidance_weight: 10.0
  trained_rtc_mode: false
```

然后运行：

```bash
uv run python examples/tron2/pi_client_rtc.py \
  --deploy-config configs/deploy/tron2_deploy.local.yaml
```

RTC client 会先 warmup 模型并填充 action queue；运行中会在短暂观测超时时等待新鲜
观测，不复用旧 obs；同时记录队列合并诊断，并可通过
`client.rtc_action_postprocess` 开启可选的动作平滑。

## 训练新的 TRON2 任务

公开任务配置建议使用 YAML 入口，而不是直接改 `src/openpi/training/config.py`：

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
```

修改 `configs/train/tron2_tasks/my_task.yaml` 后，先把 LeRobot 数据集根目录指向你
自己的数据集目录。如果 `repo_id: my_dataset`，通常需要存在
`$HF_LEROBOT_HOME/my_dataset/data/` 和 `$HF_LEROBOT_HOME/my_dataset/meta/`：

```bash
export HF_LEROBOT_HOME=/path/to/datasets
```

首次训练前先计算 normalization statistics：

```bash
uv run scripts/compute_norm_stats.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

然后启动训练：

```bash
uv run scripts/train_tron2_task.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

如需一条命令完成“先计算 norm、再训练”的本地或容器流程，可以使用公开的一站式入口。
除非传入 `--skip-norm`，它会先运行 norm 计算，再启动训练：

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --task-config configs/train/tron2_tasks/my_task.yaml \
  --exp my_task \
  --data-dir "$HF_LEROBOT_HOME" \
  --max-frames 100000
```

真实任务 YAML 已被 `.gitignore` 忽略；公开仓库只保留
`configs/train/tron2_tasks/example.yaml`。模板支持 `repo_id`、prompt、数据列名、
`action_horizon`、`state_dim`、base checkpoint 权重、输出路径，以及可选的
`prompt_from_task` 和 `rtc_training_simulated_delay`。

## 调试输出

在 YAML 中开启对应选项后，运行时生成文件会写入 `debug_images/`：

- `cam_high.jpg`
- `cam_left_wrist.jpg`
- `cam_right_wrist.jpg`
- `client.save_record: true` 时生成 `tron2_action_data.csv`
- `client.save_record: true` 时生成 `tron2_state_data.csv`
- RTC 运行会生成 `tron2_rtc_action_data.csv` 和 `tron2_rtc_state_data.csv`

这些文件只是本地诊断输出，不应提交到仓库。

## 网络部署边界

当前全部运行时网络接口——policy server/client、TRON2 机器人控制和 Bridge 观测——
仅支持授权系统接入的受控机器人局域网。不得将这些接口暴露到互联网，也不得在
不受信任或共享网络中使用。

当前传输并非都提供应用层鉴权或 TLS。不能因为 policy serving 和机器人控制链路
运行在局域网中，就把它们视为已经鉴权或加密；配置 `wss://` Bridge 端点也不会保护
其他链路或扩大受支持的信任边界。任何面向互联网、跨站点或云端的拓扑都必须在
使用前单独进行安全评审。

当前任务、prompt、RTC/训练设计、机器人映射、标定、初始化姿态和 replay 行为的
源码公开，不等于功能安全批准或真机认证，也不表示本仓库已经实现鉴权、TLS、
急停、运动限位、碰撞保护或 watchdog。

私下报告漏洞及完整部署边界见 `SECURITY.md`。

## 安全注意事项

- 真机客户端只能在受过训练的操作人员在场时运行。
- 执行 policy 前先在目标机器人上确认 `robot.init_joints` 和 `robot.init_head` 安全可达。
- 保持机器人工作空间清空，并确保急停可用。
- 首次运行时先设置较小的 `client.max_steps`，确认行为后再持续运行。
- 运行 policy 前用调试图像确认相机顺序正确。
- 本仓库不包含私有 low-level 安全控制器。

## 常见问题

- 如果 `uv sync` 或 `uv run` 很慢，检查 Python 包索引和网络访问。
- 如果 policy 加载失败，检查 `policy.config`、`policy.repo_id` 和 `policy.checkpoint_dir`。
- 如果缺少归一化统计，检查 `checkpoint_dir/assets/<policy.repo_id>/norm_stats.json`。
- 如果客户端连不上 policy server，检查 `client.policy_host` 和 `client.policy_port`。
- 如果 Bridge 观测超时，检查 `bridge.host`、TLS 设置和 Bridge 服务状态。
- 如果 legacy 模式找不到相机，检查 `camera.serial_to_name` 中的 RealSense 序列号。
- 如果机器人不运动，检查 `robot.ip`、`robot.port`、控制器状态，并确认工作空间安全后再重试。

## 第三方来源

本仓库基于 OpenPI 派生，并保留上游 OpenPI 组件。部分文件还包含来自 Big Vision、
HuggingFace Transformers、LeRobot RTC、Physical Intelligence Kinetix 和
`msgpack-numpy` 的改编代码。OpenPI commit
`e01d2290dfef823304b9a59a94b29e5945e38b2d` 是获批的 working baseline，
不表示每个组件的精确来源都已确认。路径、来源、许可证和修改状态见 `NOTICE`、
`THIRD_PARTY_NOTICES.md` 和 `MODIFICATIONS.md`。

## 贡献

贡献内容应保持在上述公开部署范围内。请不要提交私有机器人配置、真实凭据、内部
URL、客户数据、数据集、模型权重或日志。基础贡献说明见 `CONTRIBUTING.md`。

## 许可证

源码文件遵循各自的文件级源码许可证。除非另有说明，项目源码使用 `LICENSE` 中的
Apache License 2.0；第三方源码继续使用 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/`
中记录的许可证。

`LICENSE_GEMMA.txt` 按上游原文逐字节保留，作为上游模型资产条款材料。当前源码
快照不包含 Gemma 或 PaliGemma 权重、checkpoint 或模型衍生物。外部模型资产及其
衍生物适用的 Gemma Terms 需单独遵守；这些条款不会重新许可 Apache 源码，也不会
给 Apache 源码增加限制。未来发布模型资产、模型衍生物或 Hosted Service 必须重新评审。
