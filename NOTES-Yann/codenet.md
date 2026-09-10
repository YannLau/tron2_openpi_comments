# openpi 项目代码网络全览

> **openpi** — Physical Intelligence 的开源 VLA (Vision-Language-Action) 机器人操作库。
> 支持 π₀ (流匹配)、π₀-FAST (自回归)、π₀.₅ (升级流匹配 + 知识隔离) 三种架构，
> 每种均有 JAX (Flax NNX) 和 PyTorch 双实现。

---

## 一、项目总览 ── 架构分层

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户入口 / 脚本层                                 │
│  scripts/train.py │ train_pytorch.py │ serve_policy.py │ compute_norm_stats │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            配置注册中心                                      │
│        src/openpi/training/config.py  (─25 个命名 TrainConfig)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
┌─────────────────────┐ ┌─────────────────┐ ┌──────────────────────────┐
│   模型定义 (JAX)     │ │   训练流水线     │ │   推理 / 策略层          │
│ src/openpi/models/   │ │ src/openpi/     │ │ src/openpi/policies/    │
│                      │ │   training/     │ │   + serving/            │
│  model.py (基类)     │ │                 │ │                         │
│  pi0.py (π₀/π₀.₅)   │ │ data_loader.py  │ │ policy.py (推理封装)    │
│  pi0_fast.py (FAST)  │ │ optimizer.py    │ │ policy_config.py (工厂) │
│  pi0_config.py       │ │ checkpoints.py  │ │ aloha_policy.py         │
│  gemma.py (语言模型) │ │ weight_loaders  │ │ droid_policy.py         │
│  siglip.py (视觉)    │ │ sharding.py     │ │ libero_policy.py        │
│  vit.py / lora.py    │ │ utils.py        │ │ websocket_policy_server │
│  tokenizer.py        │ └─────────────────┘ └──────────────────────────┘
│  gemma_fast.py       │
│  dummy_model.py      │
└─────────────────────┘
        │                               ▲
        ▼                               │
┌─────────────────────┐       ┌──────────────────────────┐
│  PyTorch 模型       │       │  openpi-client (独立包)   │
│ models_pytorch/     │       │  packages/openpi-client/  │
│  pi0_pytorch.py     │       │                           │
│  gemma_pytorch.py   │       │  base_policy.py           │
│  preprocessing_     │       │  websocket_client_policy  │
│    pytorch.py       │       │  action_chunk_broker.py   │
│  transformers_      │       │  runtime/ (agent/env/…)   │
│    replace/         │       └──────────────────────────┘
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            共享工具层                                       │
│  src/openpi/shared/                                                        │
│  array_typing │ download.py │ normalize.py │ nnx_utils.py │ image_tools.py │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据变换管道                                       │
│              src/openpi/transforms.py                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、顶层目录结构

```
openpi/
├── pyproject.toml                  # 项目元数据 + uv 工作区配置
├── README.md                       # 项目介绍
├── CLAUDE.md                       # Claude Code 指南
├── CLAUDE.zh.md                    # 中文版指南
├── CONTRIBUTING.md                 # 贡献指南
├── LICENSE / LICENSE_GEMMA.txt     # Apache 2.0 / Gemma 许可
├── uv.lock                         # 锁定依赖版本
│
├── src/openpi/                     # ★ 核心库 (JAX 后端)
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixtures
│   ├── transforms.py               # 数据变换管道 (核心)
│   │
│   ├── models/                     # 模型架构 (JAX Flax NNX)
│   ├── models_pytorch/             # PyTorch 模型实现
│   ├── policies/                   # 平台策略适配器
│   ├── training/                   # 训练流水线
│   ├── shared/                     # 共享工具
│   └── serving/                    # WebSocket 策略服务器
│
├── packages/openpi-client/         # ★ 独立客户端包 (机器人端)
│   └── src/openpi_client/          # 通过 WebSocket 远程推理
│       ├── runtime/                # 机器人控制运行时
│       └── ...
│
├── scripts/                        # ★ 入口脚本
│   ├── train.py                    # JAX 训练
│   ├── train_pytorch.py            # PyTorch 训练
│   ├── serve_policy.py             # 策略服务
│   ├── compute_norm_stats.py       # 预计算归一化统计
│   └── train_test.py               # 集成测试
│
├── examples/                       # ★ 机器人平台示例
│   ├── aloha_real/                 # 真实 ALOHA 机器人
│   ├── aloha_sim/                  # ALOHA 仿真
│   ├── droid/                      # DROID 机器人
│   ├── libero/                     # LIBERO 基准评估
│   ├── simple_client/              # 轻量测试客户端
│   ├── ur5/README.md               # UR5 微调教程
│   ├── convert_jax_model_to_pytorch.py  # JAX→PyTorch 转换
│   ├── inference.ipynb             # 推理演示笔记本
│   └── policy_records.ipynb        # 策略记录查看器
│
├── third_party/                    # Git 子模块
│   ├── aloha/                      # Interbotix ALOHA 硬件
│   └── libero/                     # LIBERO 基准
│
├── docs/                           # 文档
│   ├── docker.md
│   ├── norm_stats.md
│   └── remote_inference.md
│
├── checkpoints/                    # 训练产物目录
├── scripts/docker/                 # Docker 部署配置
└── .github/workflows/              # CI (pre-commit + test)
```

---

## 三、核心模块详解

### 3.1 `src/openpi/models/` ── 模型架构 (JAX Flax NNX)

```
models/
├── __init__.py         # 空
├── model.py            # ★ 基类: BaseModel, BaseModelConfig, Observation
├── pi0_config.py       # ★ π₀/π₀.₅ 配置 + LoRA freeze filter
├── pi0.py              # ★ π₀/π₀.₅ 流匹配模型 (主模型)
├── pi0_fast.py         # ★ π₀-FAST 自回归模型
├── gemma.py            # Gemma 语言模型 Transformer (改编 big_vision)
├── gemma_fast.py       # FAST 专用的 Gemma 配置
├── siglip.py           # SigLIP 视觉编码器 (ViT)
├── vit.py              # 备用 ViT 实现 (SigLIP 基础)
├── lora.py             # LoRA 低秩适配层 (Einsum + FeedForward)
├── tokenizer.py        # PaligemmaTokenizer + FASTTokenizer
├── dummy_model.py      # 调试用 MLP (绕过 Gemma/SigLIP)
│
├── utils/
│   └── fsq_tokenizer.py   # 有限标量量化 (FSQ) 编解码器
│
├── pi0_test.py         # π₀ 单元测试
├── lora_test.py        # LoRA 单元测试
├── model_test.py       # 模型基类测试
└── tokenizer_test.py   # 分词器测试
```

#### 文件关系图 (模型层内部依赖)

```
pi0_config.py ──→ Pi0Config (继承 BaseModelConfig)
    │
    ├── create() ──→ pi0.Pi0 (通过延迟导入)
    │                    │
    │                    ├── gemma.Module (PaliGemma 编码器)
    │                    ├── gemma.Module (动作专家) ← 共享层但独立头
    │                    ├── siglip.Module (视觉编码器)
    │                    ├── lora.Einsum (嵌入在 gemma 中)
    │                    └── model.preprocess_observation
    │
    └── get_freeze_filter() ──→ nnx_utils.PathRegex (LoRA 冻结逻辑)

pi0_fast.py ──→ Pi0FASTConfig / Pi0FAST
    │
    ├── gemma_fast.Module
    ├── siglip.Module
    └── tokenizer.FASTTokenizer

model.py ──→ 基类 (被所有模型引用)
    │
    ├── BaseModelConfig (抽象)
    ├── BaseModel (抽象 nnx.Module)
    ├── Observation / Actions (数据结构)
    ├── preprocess_observation() (图像增强)
    └── restore_params() (orbax 恢复)
```

### 3.2 `src/openpi/models_pytorch/` ── PyTorch 实现

```
models_pytorch/
├── pi0_pytorch.py          # ★ PI0Pytorch (nn.Module) — PyTorch 主模型
├── gemma_pytorch.py        # ★ PaliGemmaWithExpertModel — 双分支架构
├── preprocessing_pytorch.py# 图像预处理 + 数据增强 (torch.compile 兼容)
│
└── transformers_replace/   # ★ 需手动复制到 site-packages/transformers 的补丁
    └── models/
        ├── gemma/
        │   ├── configuration_gemma.py  # 新增 use_adarms / adarms_cond_dim 配置
        │   └── modeling_gemma.py       # 自适应 RMSNorm (adaRMS) 实现
        ├── paligemma/
        │   └── modeling_paligemma.py   # 标准 HF PaliGemma 包装
        └── siglip/
            ├── modeling_siglip.py      # 完整 SigLIP ViT 实现
            └── check.py                # 版本校验
```

**关键架构特点 (双分支 VLA):**

```
┌──────────────────┐     ┌──────────────────────┐
│   SigLIP ViT     │     │  Gemma Language       │
│  (视觉编码器)     │────▶│  (文本嵌入 + 编码器)  │
└──────────────────┘     └──────────┬───────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │          PaliGemma LM (前缀编码)           │
              │   tokens[图像 + 语言] → 全注意力 + KV cache │
              └─────────────────────┬──────────────────────┘
                                    │ (共享 KV 前缀)
              ┌─────────────────────▼──────────────────────┐
              │       Gemma Expert (后缀去噪)              │
              │   tokens[状态 + 噪声动作 + 时间步] → 因果   │
              │   每层: 拼接 Q/K/V → 共享 MHA → 分离输出   │
              └────────────────────────────────────────────┘
                        │                  ▲
                        ▼                  │
              ┌──────────────────────────────────────┐
              │  adaRMSNorm (π₀.₅): 时间步调制归一化   │
              │  scale = Dense(cond_dim, dim)         │
              │  output = normed * (1+scale) + shift  │
              └──────────────────────────────────────┘
```

### 3.3 `src/openpi/training/` ── 训练流水线

```
training/
├── config.py               # ★ TrainConfig 数据类 + _CONFIGS 注册表
├── data_loader.py          # ★ 数据加载 (LeRobot / RLDS)
├── checkpoints.py          # 检查点管理 (orbax CheckpointManager)
├── optimizer.py            # 优化器配置 (AdamW + CosineDecay)
├── weight_loaders.py       # 预训练权重加载策略
├── sharding.py             # FSDP 分片 (2D mesh: batch × fsdp)
├── utils.py                # TrainState dataclass + 工具
├── droid_rlds_dataset.py   # DROID RLDS 数据集实现
│
├── misc/
│   ├── roboarena_config.py # RoboArena 平台配置
│   └── polaris_config.py   # PolaRiS 平台配置
│
└── data_loader_test.py     # 测试
```

#### 配置注册体系

```
_ CONFIGS = [
    # ALOHA 推理 (预训练)
    TrainConfig(name="pi0_aloha", ...),
    TrainConfig(name="pi05_aloha", ...),
    TrainConfig(name="pi0_aloha_towel", ...),
    TrainConfig(name="pi0_aloha_tupperware", ...),

    # DROID 推理
    TrainConfig(name="pi0_droid", ...),
    TrainConfig(name="pi0_fast_droid", ...),
    TrainConfig(name="pi05_droid", ...),

    # LIBERO 微调
    TrainConfig(name="pi0_libero", ...),
    TrainConfig(name="pi0_libero_low_mem_finetune", ...),
    TrainConfig(name="pi0_fast_libero", ...),
    TrainConfig(name="pi0_fast_libero_low_mem_finetune", ...),
    TrainConfig(name="pi05_libero", ...),

    # ALOHA 微调
    TrainConfig(name="pi0_aloha_pen_uncap", ...),
    TrainConfig(name="pi05_aloha_pen_uncap", ...),

    # DROID 微调
    TrainConfig(name="pi0_fast_full_droid_finetune", ...),
    TrainConfig(name="pi05_full_droid_finetune", ...),
    TrainConfig(name="pi05_droid_finetune", ...),

    # 仿真
    TrainConfig(name="pi0_aloha_sim", ...),

    # 调试
    TrainConfig(name="debug", ...),
    TrainConfig(name="debug_pi05", ...),
    TrainConfig(name="debug_restore", ...),
    TrainConfig(name="dummy_debug", ...),

    # 其他平台 (misc/)
    TrainConfig(name="roboarena", ...),
    TrainConfig(name="polaris", ...),
]
```

#### 数据流: 配置 → 数据加载

```
TrainConfig
  │
  ├── model_config: Pi0Config | Pi0FASTConfig | DummyModelConfig
  ├── data: DataConfigFactory (子类)
  │       ├── FakeDataConfig          ← 调试用随机数据
  │       ├── SimpleDataConfig         ← 通用 HuggingFace 数据集
  │       ├── LeRobotAlohaDataConfig   ← ALOHA LeRobot 格式
  │       ├── LeRobotLiberoDataConfig  ← LIBERO LeRobot 格式
  │       ├── RLDSDroidDataConfig      ← DROID RLDS 格式 (大)
  │       └── LeRobotDROIDDataConfig   ← DROID LeRobot 格式
  │
  ├── weight_loader: WeightLoader
  │       ├── NoOpWeightLoader        ← 从头训练
  │       ├── CheckpointWeightLoader  ← 从 openpi 检查点恢复
  │       └── PaliGemmaWeightLoader   ← 从官方 PaliGemma 加载
  │
  ├── optimizer: AdamW + CosineDecaySchedule
  ├── training: batch_size, steps, log, checkpoint
  └── sharding: fsdp settings


DataConfigFactory.create()
  └── DataConfig
        ├── repo_id (HuggingFace 数据集名称)
        ├── repack_transforms: Group(inputs, outputs)
        ├── data_transforms: Group(inputs, outputs)
        └── model_transforms: Group(inputs, outputs)
```

### 3.4 `src/openpi/policies/` ── 策略推理层

```
policies/
├── policy.py           # ★ Policy (推理封装) + PolicyRecorder
├── policy_config.py    # ★ create_trained_policy() 工厂
├── aloha_policy.py     # ALOHA 输入/输出变换
├── droid_policy.py     # DROID 输入/输出变换
└── libero_policy.py    # LIBERO 输入/输出变换
```

**关系图:**

```
serve_policy.py
  └── policy_config.create_trained_policy(config, checkpoint_dir)
        ├── 检测 JAX vs PyTorch (model.safetensors 文件存在?)
        ├── model.load() 或 load_pytorch()  ← 加载权重
        ├── 加载归一化统计 (norm_stats)
        └── Policy(model,
                input_transforms=[repack, 平台Inputs, 归一化, 分词...],
                output_transforms=[逆归一化, 平台Outputs...])
              ├── infer(obs_dict) → 变换链 → model.sample_actions() → 逆变换
              └── 支持 JAX jit 和 PyTorch 两种编译后端
```

### 3.5 `src/openpi/transforms.py` ── 数据变换管道

**这是将原始数据集映射到模型输入的**核心枢纽**。所有变换均实现 `DataTransformFn` 协议。**

```
变换链 (推理时):
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Repack   │──▶│ 平台特定  │──▶│ Normalize│──▶│ResizeImg │──▶│ Tokenize │──▶│PadStates │──▶ Observation
│Transform │   │Inputs     │   │(z-score) │   │(224x224) │   │(分词)    │   │&Actions  │   .from_dict()
└──────────┘   └───────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘

变换链 (训练时):
数据集键名 → Repack → 数据变换 → 归一化 → 模型变换 (含数据增强)
```

**主要变换类:**

| 类名 | 作用 |
|------|------|
| `RepackTransform` | 用路径拍平重映射键名 |
| `InjectDefaultPrompt` | 注入默认语言指令 |
| `Normalize` / `Unnormalize` | Z-score 或分位数归一化 |
| `ResizeImages` | 保持宽高比缩放 + 填充到 224x224 |
| `SubsampleActions` | 步长下采样动作序列 |
| `DeltaActions` / `AbsoluteActions` | 绝对↔增量动作互转 |
| `TokenizePrompt` | PaligemmaTokenizer 文本分词 |
| `TokenizeFASTInputs` | FASTTokenizer (含动作离散化) |
| `ExtractFASTActions` | FAST 离散 token → 连续动作 |
| `PadStatesAndActions` | 零填充到模型维度 |
| `PromptFromLeRobotTask` | 从 LeRobot 任务索引提取指令 |
| `Group(inputs, outputs)` | 分组输入/输出变换对 |
| `CompositeTransform` / `compose()` | 顺序组合 |

### 3.6 `src/openpi/serving/` ── 服务层

```
serving/
└── websocket_policy_server.py    # ★ 通过 WebSocket 提供 Policy 推理服务
```

**协议:**

```
Server (GPU)                              Client (机器人)
    │                                          │
    │  ←──── 建立 WebSocket 连接 ───────────  │
    │  ────→ meta {policy, timestamps} ────→  │
    │  ←──── msgpack[observation] ←─────────  │
    │  ────→ msgpack[action + timing] ─────→  │
    │  ←──── msgpack[observation] ←─────────  │
    │  ...  (循环)                         ... │
    │                                          │
    │  GET /healthz → 200 OK                   │
```

### 3.7 `src/openpi/shared/` ── 共享工具

```
shared/
├── array_typing.py     # jaxtyping 运行时类型检查 (@at.typecheck)
├── download.py         # GCS / HTTP 下载 + 缓存 (~/.cache/openpi)
├── image_tools.py      # resize_with_pad (JAX + PyTorch 双版本)
├── nnx_utils.py        # module_jit() 高效 JIT, PathRegex, state_map
├── normalize.py        # NormStats / RunningStats (Welford 算法)
│
├── __init__.py         # 空
├── download_test.py
├── image_tools_test.py
└── normalize_test.py
```

### 3.8 `packages/openpi-client/` ── 独立客户端包

```
packages/openpi-client/
├── pyproject.toml       # openpi-client v0.1.0, 依赖 websockets/msgpack/numpy
│
└── src/openpi_client/
    ├── __init__.py                  # __version__ = "0.1.0"
    ├── base_policy.py               # ★ BasePolicy ABC (infer + reset)
    ├── websocket_client_policy.py   # ★ WebSocket 客户端实现
    ├── action_chunk_broker.py       # ★ 动作分块调度 (高频控制)
    ├── image_tools.py               # PIL 图像缩放 + 填充
    ├── msgpack_numpy.py             # numpy 数组 msgpack 编解码
    │
    ├── runtime/                     # ★ 机器人运行时
    │   ├── agent.py                 # Agent ABC (get_action, reset)
    │   ├── agents/policy_agent.py   # PolicyAgent (桥接 BasePolicy)
    │   ├── environment.py           # Environment ABC
    │   ├── runtime.py               # ★ Runtime 主循环 (env↔agent↔subscriber)
    │   └── subscriber.py            # Subscriber ABC (记录/可视化)
    │
    ├── image_tools_test.py
    └── msgpack_numpy_test.py
```

---

## 四、数据流全景

### 4.1 训练数据流 (JAX)

```
终端
  │
  ├── uv run scripts/train.py pi05_libero --exp-name=my_exp
  │
  ▼
config.get_config("pi05_libero")     [config.py → _CONFIGS]
  │
  ▼
DataConfigFactory.create()           [config.py → DataConfig]
  │   ├── repack_transforms (重映射键名)
  │   ├── data_transforms (归一化/增量动作/默认提示)
  │   └── model_transforms (缩放图像/分词/填充)
  │
  ▼
create_data_loader()                 [data_loader.py]
  │   ├── LeRobot 数据集 (HuggingFace Datasets + PyTorch DataLoader)
  │   │   └── TransformedDataset (应用变换链)
  │   └── 或 RLDSDataLoader (for DROID, num_workers=0)
  │
  ▼
DataLoaderImpl                       [→ (Observation, Actions) 元组]
  │
  ▼
init_train_state()                   [train.py]
  │   ├── model.create() 或 model.load()  ← 含权重加载
  │   ├── 设置 FSDP sharding (jax.sharding.Mesh)
  │   └── optimizer (optax.adamw + cosine decay)
  │
  ▼
jax.jit(train_step)                  [编译训练步骤]
  │
  ▼
训练循环 (N steps)
  ├── batch = next(data_loader)
  ├── loss = model.compute_loss(batch)
  ├── grads = jax.grad(loss)
  ├── opt_state.update(grads)
  └── orbax save_state() 每 keep_every 步
```

### 4.2 推理数据流

```
serve_policy.py
  │
  ├── config.get_config("pi05_libero")
  ├── policy_config.create_trained_policy(config, checkpoint_dir)
  │   ├── 检测 JAX / PyTorch 权重
  │   ├── load_pytorch() / restore_params()
  │   ├── 加载 norm_stats (z-score 参数)
  │   └── Policy(model, 输入变换, 输出变换)
  │
  └── WebSocketPolicyServer(policy, host="0.0.0.0", port=8000)
        │
        ▼ (等待 WebSocket 连接)
        │
  ┌──────────────────────────────────────────────────────────────┐
  │                    推理循环                                   │
  │                                                              │
  │  客户端 msgpack[obs] → 服务器                                 │
  │    1. 输入变换链:                                            │
  │       obs_dict → Repack → 平台Inputs → Normalize             │
  │       → ResizeImages → TokenizePrompt → PadStatesAndActions  │
  │       → Observation.from_dict()                              │
  │    2. model.sample_actions(obs)  (10步欧拉去噪或自回归)       │
  │    3. 输出变换链:                                            │
  │       actions → Unnormalize → 平台Outputs → 裁剪              │
  │    4. msgpack[action] → 客户端                                │
  └──────────────────────────────────────────────────────────────┘
```

### 4.3 机器人部署 (Runtime)

```
┌─────────────────────────────────────────────────────────────────────┐
│  机器人端 (笔记本电脑 / NUC)                        GPU 服务器       │
│                                                                     │
│  Runtime                             WebSocketClientPolicy          │
│  ┌──────────┐                          ┌──────────────────┐        │
│  │Environment│──get_observation()──▶   │                  │        │
│  │(真实/仿真)│                        │  msgpack[obs]    │        │
│  └────┬─────┘                          │  ────────────▶   │        │
│       │                                │                  │        │
│  ┌────▼─────┐   ActionChunkBroker      │  msgpack[action] │        │
│  │  Agent    │──next chunk───────▶     │  ◀────────────   │        │
│  │(Policy)   │                        │                  │        │
│  └────┬─────┘                          │  WebSocket       │        │
│       │                                └──────────────────┘        │
│  ┌────▼─────┐                                                      │
│  │Subscriber│──on_step(obs, action)──▶ 保存日志/视频                │
│  └──────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、模块依赖关系总图

```
                              openpi (主包)
                ┌─────────────────┼──────────────────┐
                │                 │                  │
           models/          training/           policies/
                │                 │                  │
    ┌───────────┼───┐        ┌───┴───┐              │
    │       gemma.py─┼──▶lora.py    │              │
    │           │    │              │              │
    │  siglip.py    │    config.py─┼──▶models/     │
    │       │       │         │    │  policies/   │
    │  pi0.py───────┼──▶gemma │    │  transforms  │
    │       │       │   siglip│    │  shared/     │
    │  pi0_fast.py  │         │    │              │
    │       │       │    data_loader ───▶models/  │
    │  tokenizer.py │         │    │  config.py   │
    │       │       │    checkpoints──▶shared/    │
    │  model.py─────┼──▶models_pytorch            │
    └───────────┼───┘                              │
                │                                  │
           transforms.py ◀──────────────────────┘  │
                │              ▲                    │
                │    ┌────────┘                    │
                ▼    │                             │
          shared/────┘                             │
                │                                  │
                ▼                                  │
       openpi_client (独立包) ◀── policy.py ───────┘
                │
           serving/
      websocket_policy_server.py
```

---

## 六、关键文件速查表

| 文件路径 | 行数 | 角色 | 关键类/函数 |
|----------|------|------|-------------|
| `src/openpi/models/model.py` | ~340 | 模型基类 | `BaseModel`, `BaseModelConfig`, `Observation` |
| `src/openpi/models/pi0_config.py` | ~230 | π₀配置 | `Pi0Config`, `get_freeze_filter()` |
| `src/openpi/models/pi0.py` | ~420 | π₀流匹配 | `Pi0.embed_prefix/suffix`, `compute_loss`, `sample_actions` |
| `src/openpi/models/pi0_fast.py` | ~350 | π₀-FAST | `Pi0FAST`, `left_to_right_align` |
| `src/openpi/models/gemma.py` | ~900 | Gemma LM | `Module`, `Config`, 专家混合注意力 |
| `src/openpi/models/gemma_fast.py` | ~20 | FAST配置 | `Variant`, `get_config` |
| `src/openpi/models/siglip.py` | ~370 | 视觉编码 | `_Module`, `posemb_sincos_2d` |
| `src/openpi/models/vit.py` | ~430 | ViT备用 | `VisionTransformer` |
| `src/openpi/models/lora.py` | ~310 | LoRA层 | `LoRAConfig`, `Einsum`, `FeedForward` |
| `src/openpi/models/tokenizer.py` | ~390 | 分词器 | `PaligemmaTokenizer`, `FASTTokenizer` |
| `src/openpi/models/dummy_model.py` | ~120 | 调试MLP | `DummyModelConfig` |
| `src/openpi/models/utils/fsq_tokenizer.py` | ~470 | FSQ编解码 | `FsqCodebook` |
| `src/openpi/transforms.py` | ~650 | 数据变换 | `Normalize`, `TokenizePrompt`, `Group` |
| `src/openpi/policies/policy.py` | ~140 | 推理封装 | `Policy.infer()`, `PolicyRecorder` |
| `src/openpi/policies/policy_config.py` | ~100 | 策略工厂 | `create_trained_policy()` |
| `src/openpi/policies/aloha_policy.py` | ~140 | ALOHA适配 | `AlohaInputs`, `AlohaOutputs` |
| `src/openpi/policies/droid_policy.py` | ~80 | DROID适配 | `DroidInputs`, `DroidOutputs` |
| `src/openpi/policies/libero_policy.py` | ~90 | LIBERO适配 | `LiberoInputs`, `LiberoOutputs` |
| `src/openpi/training/config.py` | ~850 | 配置中心 | `TrainConfig`, `DataConfigFactory`, `_CONFIGS` |
| `src/openpi/training/data_loader.py` | ~440 | 数据加载 | `create_data_loader`, `DataLoaderImpl` |
| `src/openpi/training/checkpoints.py` | ~200 | 检查点 | `save_state`, `restore_state`, `load_norm_stats` |
| `src/openpi/training/optimizer.py` | ~100 | 优化器 | `AdamW`, `CosineDecaySchedule` |
| `src/openpi/training/weight_loaders.py` | ~200 | 权重加载 | `PaliGemmaWeightLoader`, `CheckpointWeightLoader` |
| `src/openpi/training/sharding.py` | ~160 | FSDP分片 | `make_mesh`, `fsdp_sharding` |
| `src/openpi/training/utils.py` | ~40 | 工具 | `TrainState` |
| `src/openpi/training/droid_rlds_dataset.py` | ~100 | DROID数据集 | `DroidRldsDataset` |
| `src/openpi/serving/websocket_policy_server.py` | ~140 | 策略服务 | `WebsocketPolicyServer` |
| `src/openpi/shared/array_typing.py` | ~60 | 类型检查 | `@typecheck`, `Array` |
| `src/openpi/shared/download.py` | ~100 | 下载缓存 | `maybe_download`, `get_cache_dir` |
| `src/openpi/shared/image_tools.py` | ~80 | 图像工具 | `resize_with_pad`, `resize_with_pad_torch` |
| `src/openpi/shared/nnx_utils.py` | ~80 | NNX工具 | `module_jit`, `PathRegex`, `state_map` |
| `src/openpi/shared/normalize.py` | ~120 | 归一化 | `NormStats`, `RunningStats` |
| | | | |
| `src/openpi/models_pytorch/pi0_pytorch.py` | ~460 | PyTorch主模型 | `PI0Pytorch` |
| `src/openpi/models_pytorch/gemma_pytorch.py` | ~280 | PyTorch双分支 | `PaliGemmaWithExpertModel` |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | ~170 | PyTorch预处理 | `preprocess_observation_pytorch` |
| `packages/openpi-client/src/openpi_client/base_policy.py` | ~15 | 客户端基类 | `BasePolicy` |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | ~100 | WS客户端 | `WebsocketClientPolicy` |
| `packages/openpi-client/src/openpi_client/action_chunk_broker.py` | ~50 | 动作调度 | `ActionChunkBroker` |
| `packages/openpi-client/src/openpi_client/msgpack_numpy.py` | ~80 | 序列化 | `packb`, `unpackb` |
| `packages/openpi-client/src/openpi_client/runtime/runtime.py` | ~110 | 运行时主循环 | `Runtime.run()` |
| | | | |
| `scripts/train.py` | ~660 | JAX训练入口 | `train_loop`, FSDP + optax + orbax |
| `scripts/train_pytorch.py` | ~630 | PyTorch训练入口 | DDP + AdamW + safetensors |
| `scripts/serve_policy.py` | ~120 | 服务入口 | 加载检查点 + 启动 WebSocket |
| `scripts/compute_norm_stats.py` | ~120 | 统计预计算 | `RunningStats` → `NormStats` |
| `examples/convert_jax_model_to_pytorch.py` | ~750 | 模型转换 | `slice_gemma_state_dict` |

---

## 七、架构设计要点

### 7.1 模型双专家结构 (π₀/π₀.₅)

```
同一个 Gemma Transformer 被实例化两次:
  - PaliGemma (前缀编码器): SigLIP 图像 + 语言文本 → 前缀 token
  - Action Expert (后缀去噪): 状态 + 噪声动作 + 时间步 → 后缀 token

关键: 两者共享 Transformer 层权重，但注意头 (attention head) 不共享
      → 专家的参数以命名后缀 `_1` 存储

训练时: 前缀 + 后缀 拼接 → 逐层混合注意力 (每层内拼接 Q/K/V → 共享 MHA)
推理时: 先跑前缀 → KV cache → 后缀只跑 Expert 部分 (复用缓存)
```

### 7.2 π₀ vs π₀.₅ 差异

| 特性 | π₀ | π₀.₅ |
|------|-----|-------|
| 时间步嵌入 | MLP 融合: `action_time_mlp` | adaRMSNorm 条件 |
| 状态编码 | 连续向量拼接 | 离散化后作为 token 输入 |
| 归一化 | 标准 RMSNorm | 自适应 RMSNorm (scale+shift 由时间步调制) |
| max_token_len | 48 | 200 |
| 知识隔离 | 无 | adaRMS 提供条件隔离 |

### 7.3 数据并行策略

| 框架 | 并行方式 | 实现 |
|------|---------|------|
| JAX | FSDP (全分片) | 2D Mesh (batch × fsdp), `jax.sharding.NamedSharding` |
| PyTorch | DDP (数据并行) | `torch.distributed`, `torch.nn.parallel.DistributedDataParallel` |

### 7.4 包边界

```
openpi          ← 核心库 (JAX/PyTorch, 训练, 模型, 推理)
                    ↓ 依赖
openpi-client   ← 轻量客户端 (仅 numpy/websockets，无 JAX 依赖)
                    ↓ 通过 msgpack+WebSocket 通信
```

---

## 八、文件遍历清单

### src/openpi/ (核心库)

```
src/openpi/
├── __init__.py                     # 空
├── conftest.py                     # pytest 配置
├── transforms.py                   # ★ 数据变换管道
│
├── models/                         # ★ 模型定义 (JAX Flax NNX)
│   ├── __init__.py                 # 空
│   ├── model.py                    # ★ 基类: BaseModel, BaseModelConfig, Observation
│   ├── pi0_config.py               # ★ π₀/π₀.₅ 配置
│   ├── pi0.py                      # ★ π₀/π₀.₅ 流匹配模型实现
│   ├── pi0_fast.py                 # ★ π₀-FAST 自回归模型
│   ├── gemma.py                    # Gemma Transformer (改编 big_vision)
│   ├── gemma_fast.py               # FAST 专用 Gemma 配置
│   ├── siglip.py                   # SigLIP 视觉编码器
│   ├── vit.py                      # 备用 ViT 实现
│   ├── lora.py                     # LoRA 低秩适配层
│   ├── tokenizer.py                # PaligemmaTokenizer / FASTTokenizer
│   ├── dummy_model.py              # 调试用 MLP
│   ├── utils/
│   │   └── fsq_tokenizer.py        # 有限标量量化 (FSQ)
│   ├── pi0_test.py                 # 单元测试
│   ├── lora_test.py
│   ├── model_test.py
│   └── tokenizer_test.py
│
├── models_pytorch/                 # ★ PyTorch 模型实现
│   ├── pi0_pytorch.py              # ★ PI0Pytorch (主模型)
│   ├── gemma_pytorch.py            # ★ PaliGemmaWithExpertModel
│   ├── preprocessing_pytorch.py    # 图像预处理 + 增强
│   └── transformers_replace/       # HF transformers 补丁
│       └── models/
│           ├── gemma/
│           │   ├── configuration_gemma.py  # 新增 adarms 配置
│           │   └── modeling_gemma.py       # ★ adaRMSNorm 实现
│           ├── paligemma/
│           │   └── modeling_paligemma.py   # HF PaliGemma
│           └── siglip/
│               ├── modeling_siglip.py      # SigLIP ViT
│               └── check.py               # 版本校验
│
├── policies/                       # ★ 策略推理层
│   ├── policy.py                   # Policy + PolicyRecorder
│   ├── policy_config.py            # create_trained_policy()
│   ├── aloha_policy.py             # ALOHA 适配
│   ├── droid_policy.py             # DROID 适配
│   ├── libero_policy.py            # LIBERO 适配
│   └── policy_test.py
│
├── training/                       # ★ 训练流水线
│   ├── config.py                   # ★ TrainConfig + _CONFIGS 注册表
│   ├── data_loader.py              # ★ 数据加载 (LeRobot / RLDS)
│   ├── checkpoints.py              # orbax 检查点管理
│   ├── optimizer.py                # AdamW + CosineDecay
│   ├── weight_loaders.py           # 权重加载策略
│   ├── sharding.py                 # FSDP 分片 (2D mesh)
│   ├── utils.py                    # TrainState
│   ├── droid_rlds_dataset.py       # DROID RLDS 数据集
│   ├── misc/
│   │   ├── roboarena_config.py     # RoboArena 配置
│   │   └── polaris_config.py       # PolaRiS 配置
│   └── data_loader_test.py
│
├── shared/                         # ★ 共享工具
│   ├── __init__.py                 # 空
│   ├── array_typing.py             # jaxtyping 类型检查
│   ├── download.py                 # GCS/HTTP 下载缓存
│   ├── image_tools.py              # resize_with_pad
│   ├── nnx_utils.py                # module_jit, PathRegex
│   ├── normalize.py                # NormStats / RunningStats
│   ├── download_test.py
│   ├── image_tools_test.py
│   └── normalize_test.py
│
└── serving/                        # ★ 服务层
    └── websocket_policy_server.py  # WebSocket 策略服务器
```

### scripts/ (入口脚本)

```
scripts/
├── train.py                    # ★ JAX 训练主入口
├── train_pytorch.py            # ★ PyTorch 训练主入口
├── serve_policy.py             # ★ 策略服务入口
├── compute_norm_stats.py       # 预计算归一化统计
├── train_test.py               # 集成测试
├── __init__.py                 # 空
└── docker/                     # Docker 部署
    ├── compose.yml
    ├── serve_policy.Dockerfile
    ├── install_docker_ubuntu22.sh
    └── install_nvidia_container_toolkit.sh
```

### packages/openpi-client/

```
packages/openpi-client/
├── pyproject.toml                      # 独立包配置
│
└── src/openpi_client/
    ├── __init__.py                     # __version__
    ├── base_policy.py                  # BasePolicy ABC
    ├── websocket_client_policy.py      # WebSocket 客户端
    ├── action_chunk_broker.py          # 动作分块调度
    ├── image_tools.py                  # PIL 图像工具
    ├── msgpack_numpy.py                # numpy msgpack 编解码
    ├── image_tools_test.py
    ├── msgpack_numpy_test.py
    └── runtime/
        ├── agent.py                    # Agent ABC
        ├── agents/policy_agent.py      # PolicyAgent 桥接
        ├── environment.py              # Environment ABC
        ├── runtime.py                  # Runtime 主循环
        └── subscriber.py               # Subscriber ABC
```

### examples/

```
examples/
├── aloha_real/                     # 真实 ALOHA 机器人
│   ├── main.py                     # 运行入口 (Runtime + ActionChunkBroker)
│   ├── real_env.py                 # 真实硬件接口 (Interbotix)
│   ├── env.py                      # → real_env.py 别名
│   ├── robot_utils.py              # 关节限制/插值/归位
│   ├── video_display.py            # 实时摄像头显示
│   ├── constants.py                # 硬编码常量
│   ├── convert_aloha_data_to_lerobot.py  # 数据转换
│   ├── Dockerfile / compose.yml
│   └── README.md
│
├── aloha_sim/                      # ALOHA 仿真环境
│   ├── main.py                     # 运行入口 (gym-aloha)
│   ├── env.py                      # AlohaSimEnvironment
│   ├── saver.py                    # VideoSaver 订阅者
│   ├── Dockerfile / compose.yml
│   └── README.md
│
├── droid/                          # DROID 机器人
│   ├── main.py                     # 交互式推理
│   ├── convert_droid_data_to_lerobot.py  # 数据转换
│   ├── compute_droid_nonidle_ranges.py   # 空闲过滤
│   ├── README.md                   # 推理说明
│   └── README_train.md             # 训练说明
│
├── libero/                         # LIBERO 基准
│   ├── main.py                     # 评估入口
│   ├── convert_libero_data_to_lerobot.py # 数据转换
│   ├── Dockerfile / compose.yml
│   └── README.md
│
├── simple_client/                  # 轻量测试客户端
│   ├── main.py                     # 延迟基准
│   ├── Dockerfile / compose.yml
│   └── README.md
│
├── ur5/                            # UR5 微调教程
│   └── README.md
│
├── convert_jax_model_to_pytorch.py # JAX→PyTorch 检查点转换
├── inference.ipynb                 # 推理笔记本
└── policy_records.ipynb            # 策略记录查看
```

---

> 生成日期: 2026-07-10
> 基于对项目仓库的全面代码分析。
