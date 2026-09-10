# Mujoco-aloha 仿真命令

uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=my_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim

uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=yann_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim    

MUJOCO_GL=egl uv run examples/aloha_sim/main.py

  # 测试默认随机位置（等效于不指定）
  MUJOCO_GL=egl uv run examples/aloha_sim/main.py

--args.box-pose 0.2 0.5 0.05 1 0 0 0

  # 测试 Cube 放在右侧远处 (x=0.4, y=0.3)
 MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.40 0.30 0.05 1 0 0 0

  # 测试 Cube 放在左侧 (x=0.05, y=0.55)  
MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.05 0.55 0.05 1 0 0 0

  # 测试 Cube 放在正前方远处 (x=0.25, y=0.50)
MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.25 0.50 0.05 1 0 0 0


----------------------------------

# single_data_test_lerobotv2.1_video_tron2

## 开启wandb
export WANDB_API_KEY="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"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  uv run scripts/train.py pi05_tron_single_data_lora --exp-name=tron2_single_data --overwrite --wandb_enabled --batch_size=1

## 关闭wandb
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  uv run scripts/train.py pi05_tron_single_data_lora --exp-name=tron2_single_data --overwrite --no-wandb_enabled --batch_size=1

-----------------------------------


# 并行智算云中的运行命令

export HF_LEROBOT_HOME="/root/shared-nvme/data/"

uv run scripts/compute_norm_stats.py --config-name=<config_name>

export WANDB_MODE=disabled

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tron_all_data_lora --exp-name=tron2lora --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tron_all_data_lora --exp-name=tron2lora --overwrite --save_interval=10000


# 虚假推理

要先将norm_stat.json放入对应的模型权重的assets文件夹中。保证路径对

### 启动服务器


> 要在config.py关闭lora参数,否则基础模型加载参数后会报错，因为对模型字典和权重参数字典对不上
uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi05_tron_fake_infer \
      --policy.dir=/home/punk/yann_repo/para_check_pi0.5/yann_paras/checkpoint/openpi-assets/checkpoints/pi05_base/

### 启动fake_client.py

uv run examples/tron2/fake_client.py

> Connecting to policy server at 127.0.1.1:8000 ...

============================================================
  Server Metadata (from handshake)
============================================================


########################################
  Round 1 / 3
########################################

============================================================
  Input (fake observation)
============================================================
  image: {
  cam_high:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_left_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_right_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
}
  state: np.ndarray(shape=[16], dtype=float64)  range=[-0.3290, 0.9622]
  prompt: "Put the banana on the plate."

  Inference time: 22.181s (22180.9ms)

============================================================
  Output (policy result)
============================================================
  actions: np.ndarray(shape=[50, 16], dtype=float64)  range=[-1.8798, 0.6732]
  policy_timing: {
  infer_ms:   21914.497724967077
}
  server_timing: {
  infer_ms:   22178.60574304359
}

  >>> actions shape: (50, 16)
  >>> actions range: [-1.8798, 0.6732]
  >>> first action (left arm): [ 5.11150477e-01  6.65256990e-02  4.37881821e-01 -1.87795271e+00
  4.09504215e-01 -3.05686873e-01 -1.68795508e-01  5.36552846e-07]
  >>> first action (right arm): [-0.47939466  0.40346096  0.67215568 -1.06505298  0.17049596 -0.32464477
  0.03941871  0.46218226]
  >>> last action (left arm):  [ 5.44195260e-01  6.89800976e-02  4.38519428e-01 -1.87977385e+00
  4.18252193e-01 -2.77481827e-01 -1.70698846e-01  6.42429054e-07]
  >>> last action (right arm):  [-0.44808159  0.4019612   0.61788835 -1.13287978  0.25521645 -0.31491819
  0.03876972  0.46779538]


########################################
  Round 2 / 3
########################################

============================================================
  Input (fake observation)
============================================================
  image: {
  cam_high:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_left_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_right_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
}
  state: np.ndarray(shape=[16], dtype=float64)  range=[-0.4350, 0.8880]
  prompt: "Put the banana on the plate."

  Inference time: 0.102s (102.3ms)

============================================================
  Output (policy result)
============================================================
  actions: np.ndarray(shape=[50, 16], dtype=float64)  range=[-1.9870, 0.7020]
  policy_timing: {
  infer_ms:   80.50495496718213
}
  server_timing: {
  infer_ms:   100.81281402381137
  prev_total_ms:   22181.537256983574
}

  >>> actions shape: (50, 16)
  >>> actions range: [-1.9870, 0.7020]
  >>> first action (left arm): [ 1.78417674e-01  3.02639243e-01  2.34233472e-01 -1.96942630e+00
  3.37379585e-01  3.25869044e-01 -1.44089811e-01  9.85703290e-07]
  >>> first action (right arm): [ 0.37196807  0.26336202  0.12967883 -0.99125884 -0.51332274  0.7002748
  0.45767059  0.47105143]
  >>> last action (left arm):  [ 1.62273247e-01  2.96947435e-01  2.23413165e-01 -1.98704847e+00
  3.67258642e-01  1.88396165e-01 -1.45675590e-01  9.13216829e-07]
  >>> last action (right arm):  [ 0.29117605  0.2757726   0.05756434 -1.02285739 -0.45734798  0.69794239
  0.45690254  0.475845  ]


########################################
  Round 3 / 3
########################################

============================================================
  Input (fake observation)
============================================================
  image: {
  cam_high:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_left_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
  cam_right_wrist:   np.ndarray(shape=[3, 224, 224], dtype=uint8)  range=[0, 255]
}
  state: np.ndarray(shape=[16], dtype=float64)  range=[-0.4908, 0.4773]
  prompt: "Put the banana on the plate."

  Inference time: 0.099s (99.0ms)

============================================================
  Output (policy result)
============================================================
  actions: np.ndarray(shape=[50, 16], dtype=float64)  range=[-1.6979, 0.4863]
  policy_timing: {
  infer_ms:   77.85871997475624
}
  server_timing: {
  infer_ms:   98.29865698702633
  prev_total_ms:   203.4440059796907
}

  >>> actions shape: (50, 16)
  >>> actions range: [-1.6979, 0.4863]
  >>> first action (left arm): [ 1.85700872e-01  6.14383970e-02  4.86338692e-01 -1.69532309e+00
 -2.60233454e-01 -1.23267712e-01  2.39855486e-01  8.59188914e-07]
  >>> first action (right arm): [ 0.17701218 -0.77750841  0.24924973 -1.41858832 -0.03941774 -0.17910532
 -0.09490627  0.46416378]
  >>> last action (left arm):  [ 1.89629279e-01  6.07161150e-02  4.84338827e-01 -1.69719307e+00
 -2.62013349e-01 -1.19871079e-01  2.40184202e-01  8.20352197e-07]
  >>> last action (right arm):  [ 0.18319622 -0.7557481   0.26705241 -1.39761515 -0.03375291 -0.17453299
 -0.09509176  0.46906775]

Done. All rounds completed successfully.

----------


# 真机部署指南

# 帮我检查 @examples/tron2/real_env.py 中的 def _get_images 方法是否受 Tron2Config() 中的 ip、port 影响？我追踪了很久代码，没发现，它从机器人的ip链接中获取图像，更像是在获取运行代码的本机的摄像头中获取图像的，因此我怀疑 examples/tron2/pi_client.py 是否应该运行在 tron2的机器人内部的linux上呢？

现在我已经完整地追踪了整条调用链。以下是完整分析：

---

## 结论 1：`_get_images()` **完全不依赖** `Tron2Config.ip` 和 `port`

两者的数据流是完全独立的两个通道：

| 组件                                | 通信方式               | 目标                                | 使用 `ip`/`port`？ |
| ----------------------------------- | ---------------------- | ----------------------------------- | ------------------ |
| **`Tron2`** (机器人控制)            | WebSocket              | 机器人控制器 (10.192.1.2:5000)      | ✅ 是              |
| **`MultiCameraManager`** (图像采集) | **USB (pyrealsense2)** | **本机物理连接的 RealSense 摄像头** | ❌ 否              |

追踪一下 `_get_images()` 的实际调用链：

```
Tron2Env._get_images()
  → self.camera_manager.get_all_latest_frames()
    → MultiCameraManager (realsense_image.py)
      → 初始化的方式是：
        raw_config → MultiCameraManager.from_config()    ← 用摄像头序列号映射
        或默认     → MultiCameraManager()                ← 用默认序列号映射
      → detect_cameras() 通过 pyrealsense2 检测 **USB连接的本地摄像头**
        rs.context().query_devices()  ✦  通过 USB 枚举设备序列号
      → setup_pipelines() 配置 RealSense 管道
        pipeline.start(config)  ✦  启动本地 USB 摄像头采集
      → _capture_loop() 后台循环
        pipeline.wait_for_frames()  ✦  从本地 USB 摄像头获取帧
```

所有 RealSense API 调用都是通过 **USB** 与本机连接的摄像头通信，**与 `Tron2Config.robot_ip` 和 `port` 没有任何关系**。`ip`/`port` 仅在 `Tron2` 类的 WebSocket 连接中使用（发给机器人控制器发送关节指令和接收关节状态反馈）。

---

## 结论 2：`pi_client.py` 应该运行在**连接了 RealSense 摄像头的机器上**

### 从摄像头物理位置看

三个摄像头在代码中的定义：
- **`head_camera_image`** — 安装在机器人**头部**
- **`left_wrist_image`** — 安装在机器人**左腕**
- **`right_wrist_image`** — 安装在机器人**右腕**

这些摄像头**物理上在机器人身上**。它们的 USB 线要连到运行 `pi_client.py` 的电脑上。

### 从代码依赖看

在 `real_env.py:272-284` 中：

```python
def _init_camera(self):
    from realsense_image import MultiCameraManager  # ← 本地 RealSense 驱动
    camera_manager = MultiCameraManager(
        max_queue_size=self.config.camera_config.max_queue_size
    )
    camera_manager.start_capture()                   # ← 启动本地摄像头采集
    time.sleep(3.0)                                  # ← 等待预热
    return camera_manager
```

在 `realsense_image.py:228-234`：

```python
def detect_cameras(self):
    ctx = rs.context()              # ← pyrealsense2 本地上下文
    devices = ctx.query_devices()  # ← 枚举本地 USB 设备
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
```

**RealSense 摄像头必须通过 USB 连接到运行代码的机器**，不能通过网络获取图像。

### 推荐的部署架构

```
┌─────────────────────────────────────┐
│  Tron2 机器人本体                     │
│  ┌────────┐  ┌────────────────────┐  │
│  │ 机器人  │  │ 头/腕 RealSense  │  │
│  │ 控制器  │  │ 摄像头 (USB)       │  │
│  │(10.192. │  └────────┬───────────┘  │
│  │ 1.2:5000)│           │ USB        │
│  └────┬────┘           │            │
└───────┼────────────────┼────────────┘
        │ WebSocket      │
        ▼                ▼
┌──────────────────────────────────────┐
│  机器人载板 Linux PC                  │
│  (运行 pi_client.py)                  │
│                                      │
│  ┌─────────────────────────────────┐│
│  │  Tron2Env                       ││
│  │  ├─ robot_utils.Tron2 ───────── et│
│  │  │   (WebSocket → 10.192.1.2:5000)│
│  │  └─ MultiCameraManager ──────────│
│  │      (USB → 本地 RealSense)     │|
│  └─────────────────────────────────┘│
│                    │ WebSocket      │
│                    ▼                │
│  ┌─────────────────────────────────┐│
│  │ WebsocketClientPolicy ──────────││
│  │  → GPU 服务器 (端口 8000)       ││
│  └─────────────────────────────────┘│
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  GPU 服务器                          │
│  (运行 serve_policy.py)              │
│  ┌─────────────────────────────────┐│
│  │  π₀ 模型推理                    ││
│  │  (不需要连接机器人/摄像头)       ││
│  └─────────────────────────────────┘│
└──────────────────────────────────────┘
```

### 总结

1. **`_get_images()` 完全不受 `Tron2Config.ip`/`port` 影响** — 图像来自本地 USB RealSense 摄像头，与控制机器人的网络连接是两条独立的路径

2. **`pi_client.py` 应该运行在连接了 RealSense 摄像头的机器上**（通常是 Tron2 机器人的载板 Linux），因为它需要：
   - **物理 USB 连接**到三个 RealSense 摄像头来获取图像
   - **网络连接**到机器人控制器 (10.192.1.2:5000) 下发关节指令
   - **网络连接**到 GPU 服务器 (port 8000) 进行策略推理

   它**不需要**运行在机器人控制器（10.192.1.2）上，那个通常是实时嵌入式系统，只负责底层的电机控制。