"""
π₀ (Pi0) 模型的核心实现。

本文件实现了 Physical Intelligence 的 π₀ 模型，这是一个基于流匹配（Flow Matching）
的视觉-语言-动作（VLA, Vision-Language-Action）模型。

核心架构：
- PaliGemma（视觉语言模型）作为 backbone，负责理解图像和语言指令
- 动作专家（Action Expert）通过流匹配从噪声生成机器人动作
- 支持 RTC（Real-Time Chunking，实时分块）以实现低延迟的实时控制

流匹配简介（Flow Matching）：
流匹配是一种生成模型，它学习一个从噪声分布到目标分布的向量场 v_t。
训练时，我们通过插值构造带噪样本 x_t = t*noise + (1-t)*actions，目标是
预测 v_t = noise - actions（从带噪样本指向干净样本的方向）。
推理时，从纯噪声开始，用学到的向量场逐步去噪，最终得到干净的动作序列。

时间约定（重要！）：
- 本代码遵循扩散模型文献中更常见的约定：t=1 是纯噪声，t=0 是目标（干净动作）
- 这与 π₀ 论文中的约定相反（论文中 t=0 是噪声，t=1 是目标）
- dt 为负值（从 t=1 向 t=0 推进）

RTC（Real-Time Chunking）概述：
机器人在执行动作时需要低延迟响应。如果等模型生成完所有动作再执行，
延迟会很高。RTC 将动作分成多个"块"（chunk），让机器人可以先执行前几个
动作，同时模型在后台继续生成后续动作。有两种实现方式：
1. 推理时 RTC：使用 VJP 计算前缀引导修正（Kinetix 方法），训练时无需特殊处理
2. 训练时 RTC：在训练时就模拟延迟，推理时直接用（更快，但需要专门训练）

本文件的阅读顺序（新手建议）：
1. 先读上方「流匹配」与「时间约定」——不理解 t 的方向（t=1 噪声 → t=0 目标），后面全是错的；
2. 读 `make_attn_mask` 与 `posemb_sincos`：两个独立小工具，理解「注意力块（前缀双向 / 动作因果）」和「时间编码」两个概念；
3. 读 `Pi0.__init__`：了解模型由「PaliGemma（视觉+语言，VLM 专家 + 动作专家共享参数）」+「若干投影层」组成；
4. 读 `embed_prefix` / `embed_suffix`：理解哪些 token 进前缀（条件，双向注意力、可缓存 KV）、哪些进后缀（要预测的动作，因果注意力）；
5. 读 `compute_loss`：训练时如何构造带噪样本 x_t、让模型学向量场 v_t（先跳过 RTC 分支）；
6. 最后读 `sample_actions` 及两个 `_sample_actions_*` 辅助方法：推理时如何从噪声去噪，以及 RTC 的两种实现。
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.rtc.rtc_processor import get_prefix_weights
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """构建注意力掩码（Attention Mask），改编自 big_vision。

    这个函数是实现灵活注意力模式的核心工具。它通过一个简单的 bool 数组 `mask_ar`
    来控制哪些 token 之间可以互相看到（attend）。

    核心思想：
    - 当 mask_ar 在某位置为 True 时，表示该位置的 token 开始了一个新的"注意力块"
    - 同一个注意力块内的所有 token 共享相同的注意力规则（cumsum 值相同）
    - token 只能 attend 到 cumsum 值 ≤ 自己 cumsum 值的 token（即前面的块和同块）

    注意力模式举例：

      [[1 1 1 1 1 1]]: 纯因果注意力（causal attention）。
          每个 token 只能看到自己和前面的 token。

      [[0 0 0 1 1 1]]: 前缀-LM 注意力（prefix-lm attention）。
          前 3 个 token 可以互相看到（双向注意力），
          后 3 个 token 是因果注意力（只能看前面和自己）。
          第一个元素也可以是 1，行为不变。

      [[1 0 1 0 1 0 0 1 0 0]]: 4 个块之间的因果注意力。
          同一块内的所有 token 可以互相看到，
          同时每个块也能看到前面所有块的全部 token。

    在 π₀ 中的应用：
    - 前缀（图像 + 语言 token）之间是双向注意力（ar_mask 全为 False）
    - 动作 token 对前缀是因果注意力（ar_mask 第一个为 True）
    - 动作 token 之间是因果注意力（ar_mask 后续为 False）

    Args:
      input_mask: bool[B, N]，标记哪些位置是真实输入（True），哪些是 padding（False）
      mask_ar: bool[?B, N]，注意力重置掩码。True 表示开始一个新的注意力块，
          该 token 及其之后不能通过 False 看到前面的 token

    Returns:
      bool[B, N, N] 的注意力掩码，attn_mask[b, q, k] = True 表示 query token q
      可以 attend 到 key token k
    """
    # 将 mask_ar 广播到与 input_mask 相同的形状
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # cumsum: 给每个注意力块分配一个递增的 ID
    # 每次遇到 True 时 cumsum 增加 1，同一块内 False 保持相同的 cumsum 值
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 核心规则：query 的 cumsum 值 ≥ key 的 cumsum 值 → 可以 attend
    # [:, None, :] 是 query 维度，[:, :, None] 是 key 维度
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # padding 位置的 token 不能作为 query 也不能作为 key
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """计算正弦-余弦位置编码（Sinusoidal Positional Embedding）。

    这是一个标准的位置编码方法（类似 Transformer 中的位置编码），
    用于将标量位置（如时间步 t）编码为高维向量。

    工作原理：
    - 使用对数间隔的频率，从高频（min_period）到低频（max_period）
    - 每个频率对应一对 (sin, cos)，共同构成 embedding_dim 维的向量
    - 这样不同频率的 sin/cos 能捕获不同尺度的位置信息

    在 π₀ 中的应用：
    将扩散/流匹配的时间步 t（0~1 的标量）编码为可被模型理解的高维向量，
    让模型知道当前处于去噪过程的哪个阶段。

    Args:
      pos: 标量位置值，形状 (b,)，例如时间步 t
      embedding_dim: 输出的编码维度（必须是偶数，因为 sin/cos 成对出现）
      min_period: 最小周期（对应最高频率），控制对细微变化的敏感度
      max_period: 最大周期（对应最低频率），控制对全局位置的感知

    Returns:
      形状为 (b, embedding_dim) 的位置编码向量，
      前一半是 sin 值，后一半是 cos 值
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 在 [0, 1] 之间均匀采样，控制不同维度的频率
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    # 对数间隔的周期：从 min_period 到 max_period
    # 例如 min_period=4e-3, max_period=4.0 → 周期从 0.004 到 4.0
    period = min_period * (max_period / min_period) ** fraction
    # 外积计算：pos[i] * (2π / period[j])
    # 结果形状 (b, embedding_dim // 2)
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 拼接 sin 和 cos，得到 (b, embedding_dim) 的编码
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """π₀ 模型：基于流匹配的视觉-语言-动作（VLA）模型。

    模型架构概览：
    ┌─────────────────────────────────────────────────────────┐
    │                     PaliGemma（Backbone）                │
    │  ┌──────────────┐  ┌──────────────────────────────────┐ │
    │  │  SigLIP       │  │  Gemma LLM                      │ │
    │  │  (图像编码器)  │  │  (前缀: VLM + 后缀: 动作专家)    │ │
    │  │  图像→Token   │  │  输入Token→预测向量场 v_t        │ │
    │  └──────────────┘  └──────────────────────────────────┘ │
    │                           ↑                             │
    │              共享参数，通过 adarms 切换专家              │
    └─────────────────────────────────────────────────────────┘

    两代架构（由 config.pi05 控制）：

    π₀（pi05=False，原始版本）：
    - 使用独立的 state token + 动作 token
    - 时间信息通过 MLP 与动作 token 混合
    - Gemma 中的 adarms（自适应 RMS）全部禁用

    π₀.₅（pi05=True，改进版本）：
    - 移除了独立的 state token
    - 时间信息通过 adarms（自适应 RMS Norm）注入 Gemma
    - adarms 让时间条件能非线性地调节每一层的归一化参数
    - 动作 token 直接传入，不做时间混合
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        """初始化 π₀ 模型。

        Args:
            config: π₀ 的配置对象，包含模型变体、动作维度、预测长度等参数
            rngs: JAX 随机数生成器，用于初始化模型参数
        """
        # 调用父类，保存 action_dim（动作维度，如 7 自由度机械臂 = 7）、
        # action_horizon（预测的动作步数，如 50）、max_token_len（最大 token 数）
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # 是否为 π₀.₅ 版本

        # 获取 Gemma 的配置（两种变体共享相同的 width 等基础参数，但可以独立配置）
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ========== 构建 Gemma LLM（大语言模型主干） ==========
        # 使用两个配置创建 Gemma 模块：
        # - paligemma_config: VLM 前缀部分的配置（处理图像+语言）
        # - action_expert_config: 动作专家的配置（处理/生成动作）
        # 两者共享参数，但可以通过 adarms 做任务特定的条件调节
        # TODO: 未来直接重写为 NNX，目前使用桥接层兼容旧代码
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,  # 嵌入层的数据精度（如 bfloat16）
                adarms=config.pi05,         # π₀.₅ 启用 adarms（自适应 RMS）
            )
        )
        # 延迟初始化：在第一次调用时推断参数形状
        # π₀.₅: 前缀禁用 adarms，后缀启用 → [False, True]
        # π₀:   全部禁用 adarms → [False, False]
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # ========== 构建 SigLIP 图像编码器 ==========
        # SigLIP 将图像编码为固定数量的视觉 token
        # variant="So400m/14": 使用 SigLIP So400m 变体，patch size 14
        # pool_type="none": 不池化，保留所有 patch token（而非只取 [CLS]）
        # scan=True: 使用扫描式执行以节省显存
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 输出维度匹配 Gemma 的 hidden dim
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # 用一张假图像进行延迟初始化，推断参数形状
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # 将 LLM 和图像编码器组合成 PaliGemma
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ========== 动作投影层 ==========
        # action_in_proj: 将原始动作（如 7 维关节角）投影到 Gemma 的隐空间维度
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        # 根据版本选择不同的时间注入方式
        if config.pi05:
            # π₀.₅: 时间通过 adarms 注入
            # time_mlp_in → swish → time_mlp_out → swish → 作为 adarms 的条件向量
            # 这个条件向量会调节 LLM 每一层的 RMS Norm 参数
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # π₀: 时间通过 MLP 与动作 token 直接混合
            # state_proj: 将状态（当前关节角）投影到隐空间
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # action_time_mlp: 将动作 token 和时间 token 拼接后混合
            # 输入是 2*width（动作 + 时间各占一半），输出是 width
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # action_out_proj: 将 LLM 输出的隐空间向量投影回原始动作空间
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 确定性模式标记：训练时 model.train() 设为 False（启用 dropout），
        # 推理时 model.eval() 设为 True（禁用 dropout）
        # 这里初始化为 True 是因为创建后默认处于 eval 模式
        self.deterministic = True

        # ========== RTC（Real-Time Chunking）配置 ==========
        # 这些配置是 Python 常量，在 module_jit 时被冻结（如同 self.pi05）
        # 注意：它们不是 JAX 数组，不能作为 jax.jit 的追踪值传递

        # rtc_prefix_schedule: 前缀注意力衰减策略
        # "exp" = 指数衰减：越靠后的动作对前缀的依赖越弱
        self.rtc_prefix_schedule = "exp"

        # rtc_max_guidance_weight: RTC 引导权重的最大值
        # 防止去噪步长过大的裁切值
        self.rtc_max_guidance_weight = 10.0

        # rtc_simulated_delay: 训练时 RTC 的模拟延迟
        # None 表示不使用训练时 RTC（使用推理时 VJP 方法）
        # 正数表示最大延迟步数
        self.rtc_simulated_delay = config.rtc_training_simulated_delay

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """构建前缀嵌入（Prefix Embedding）：图像 + 语言指令的 token 序列。

        前缀是模型的"条件输入"——它提供了场景理解和任务描述，
        但不包含任何动作信息。前缀 token 在去噪过程中保持不变，
        可以预先计算并缓存 KV cache 以加速推理。

        前缀 token 之间的注意力模式：
        - 图像 token 之间：双向注意力（可以互相看到）
        - 语言 token 之间：双向注意力
        - 图像和语言 token 之间：双向注意力（可以互相看到）
        即整个前缀内部是双向注意力，没有因果限制。

        Returns:
            tokens: 形状 (b, s, emb) 的 token 序列
            input_mask: 形状 (b, s) 的有效性掩码（True=有效, False=padding）
            ar_mask: 形状 (s,) 的注意力重置掩码（前缀内部全 False=双向注意力）
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # ========== 1. 嵌入图像 ==========
        # 遍历所有相机/图像源（如 front, left, right, wrist 等）
        for name in obs.images:
            # SigLIP 编码图像 → 输出 (b, num_patches, embed_dim) 的视觉 token
            # train=False：只是让图像编码器前向传播保持确定性（关闭 dropout 等随机操作）——
            #   图像增强已在 preprocess_observation 阶段做完，编码器内部无需随机性。
            #   注意：这**不代表**冻结权重！梯度仍会反向传播，pi0 微调默认是端到端训练
            #   整个 PaliGemma（含图像编码器）。真正的参数冻结由配置里的 get_freeze_filter 决定。
            # 返回值 (image_tokens, _) 中 `_` 是中间层激活（供可视化/分析用），这里直接丢弃。
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            # 图像 mask：标记哪些位置是有效图像 token
            # obs.image_masks[name] 形状 (b,) → 复制到每个 patch
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 之间是双向注意力 → ar_mask 全部为 False
            # 注意：ar_mask 维度是 (s,)，跨所有 token 共享
            ar_mask += [False] * image_tokens.shape[1]

        # ========== 2. 嵌入语言（分词后的文本指令） ==========
        if obs.tokenized_prompt is not None:
            # 将 token ID 序列通过 Gemma 的嵌入层转换为向量
            # method="embed": 只做嵌入，不经过 Transformer 层
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 语言 token 与图像 token 之间也是双向注意力
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 沿序列维度拼接所有前缀 token
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        """构建后缀嵌入（Suffix Embedding）：状态 + 噪声动作 + 时间步。

        后缀是模型需要"去噪"的部分——包含带噪声的动作序列和时间条件。
        模型通过 LLM 处理后缀，预测从噪声指向干净动作的向量场 v_t。

        Args:
            obs: 观测数据（包含当前机器人状态）
            noisy_actions: 带噪声的动作序列，形状 (b, H, action_dim)
                - H = action_horizon（预测步数，如 50）
                - action_dim = 动作维度（如 7 自由度关节角）
            timestep: 时间步参数。
                - 标准路径: 形状 (b,)，每个 batch 元素一个时间标量
                - 训练时 RTC: 形状 (b, H)，每个动作 token 可以有不同的时间

        Returns:
            Tuple of:
            - tokens: 形状 (b, s, emb) 的后缀 token 序列
            - input_mask: 形状 (b, s) 的有效性掩码
            - ar_mask: 形状 (s,) 的注意力重置掩码
              （动作 token 不能 attend 前缀，动作之间是因果注意力）
            - adarms_cond: 时间条件向量（π₀.₅ 用）或 None（π₀ 用）
        """
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # ========== π₀: 添加独立的状态 token ==========
            # 将当前机器人状态（关节角等）编码为单个 token
            # state_proj: (b, action_dim) → (b, embed_dim) → 添加序列维度 → (b, 1, embed_dim)
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # ar_mask 开始为 True：标志着前缀部分的注意力块结束
            # 图像/语言 token 不能 attend 到这个 state token 及后面的动作 token
            ar_mask += [True]

        # ========== 动作 token 投影 ==========
        # 将动作从原始空间投影到 LLM 的隐空间
        # (b, H, action_dim) → (b, H, embed_dim)
        action_tokens = self.action_in_proj(noisy_actions)

        # ========== 时间步编码 ==========
        # 使用正弦-余弦位置编码将时间标量转换为高维向量
        # min_period=4e-3（高频）到 max_period=4.0（低频），对 [0, 1] 范围的时间敏感
        if timestep.ndim == 1:
            # 标准路径：每个 batch 一个标量时间 → (b, embed_dim)
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        else:
            # 训练时 RTC：每个 token 有自己的时间（b, H）
            # reshape 为 (b*H,) → 编码 → reshape 回 (b, H, embed_dim)
            batch_size = timestep.shape[0]
            emb_dim = self.action_in_proj.out_features
            time_flat = timestep.reshape(-1)  # (b*H,)
            time_emb_flat = posemb_sincos(time_flat, emb_dim, min_period=4e-3, max_period=4.0)  # (b*H, emb)
            time_emb = time_emb_flat.reshape(batch_size, self.action_horizon, emb_dim)  # (b, H, emb)

        # ========== 时间条件注入（两种策略） ==========
        # 注意：pi05 模式下本函数完全用不到 obs —— 状态（state）在数据变换阶段已被分词
        # 拼进 prompt（见 transforms.TokenizePrompt 的 discrete_state_input=True），由前缀
        # embed_prefix 当作语言 token 处理。这正是 pi0_config 里
        # 「pi05：state 作为离散语言 token」这一架构区别。
        if self.pi05:
            # π₀.₅: 通过 adarms（自适应 RMS Norm）注入时间条件
            # adarms 是 Gemma 每一层 RMS Norm 的调节向量，让时间信息
            # 非线性地影响模型各层的归一化行为
            # 使用标量时间（如果是 per-token time，取均值）
            t_for_ada = timestep if timestep.ndim == 1 else jnp.mean(timestep, axis=-1)
            # 时间 → sin/cos 编码 → MLP → swish → MLP → swish → adarms 条件向量
            ada_emb = posemb_sincos(t_for_ada, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            ada_emb = self.time_mlp_in(ada_emb)
            ada_emb = nnx.swish(ada_emb)
            ada_emb = self.time_mlp_out(ada_emb)
            ada_emb = nnx.swish(ada_emb)
            # 动作 token 不做额外处理，直接传入 LLM
            action_expert_tokens = action_tokens
            adarms_cond = ada_emb
        else:
            # π₀: 通过 MLP 将时间 token 与动作 token 混合
            # 1. 将时间编码复制到每个动作位置（如果还没是 per-token）
            if timestep.ndim == 1:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            else:
                time_tokens = time_emb  # 已经是 (b, H, emb)
            # 2. 拼接动作和时间 → MLP → 混合后的 token
            # 输入维度: 2 * embed_dim（动作一半，时间一半）
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None  # π₀ 不使用 adarms

        # ========== 组装后缀 ==========
        tokens.append(action_expert_tokens)
        # 动作 token 全部有效（无 padding）
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        # 注意力规则（重要！）：
        # - 第一个动作 token 的 ar_mask 为 True：表示前缀不能 attend 到后缀
        # - 后续动作 token 的 ar_mask 为 False：动作 token 之间是因果注意力
        #   即后面的动作可以看到前面的动作，反之不行
        # 整体效果：前缀 → [state] → 动作₀ → 动作₁ → ... → 动作_{H-1}
        #          ↑前缀不能看到后面   ↑动作间因果
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # 拼接所有 token
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算流匹配（Flow Matching）训练损失。

        流匹配的核心思想：
        1. 在噪声和真实动作之间插值构造训练样本
        2. 让模型学习从噪声指向真实动作的向量场 v_t
        3. 推理时沿着 v_t 逐步去噪即可从噪声恢复出干净动作

        具体步骤（标准流匹配）：
        x_t = t * noise + (1 - t) * actions    （带噪样本：线性插值）
        u_t = noise - actions                    （目标向量场：指向干净样本）
        v_t = model(x_t, t)                      （模型预测的向量场）
        loss = MSE(v_t, u_t)                     （均方误差）

        RTC 训练模式：
        当 rtc_simulated_delay > 0 时，模拟推理时的延迟场景：
        - 随机采样一个延迟 d
        - 前 d 个动作位置被当作"已提交"（time=0，即干净动作）
        - 模型只需预测剩余位置的动作
        - 这让模型学会在给定已提交前缀的情况下生成连贯的后续动作

        Args:
            rng: 随机数种子
            observation: 观测数据（图像、状态、文本指令）
            actions: 真实动作序列，形状 (*b, H, action_dim)
            train: 是否为训练模式（影响数据增强）

        Returns:
            形状 (*b, H) 的损失值（每个 batch 元素的每个动作位置的损失）
        """
        # 整个函数分两大块，模型前向传播完全一样，只有「时间」的构造方式不同：
        #   1. RTC 训练分支（if ... return）：每个动作位置一个时间，前 delay 步设为 t=0
        #      （已提交/干净），并把这些位置的损失清零；
        #   2. 标准路径：整个 batch 共享一个标量时间 t。
        # 只想理解基础流匹配的新手，建议先跳过 RTC 分支，直接读后半段标准路径。
        # 分割随机数
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        # 预处理观测（数据增强等）
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # ========== 流匹配的核心：构造带噪样本 ==========
        batch_shape = actions.shape[:-2]  # 去除最后两维 (H, action_dim)，可能是 (b,) 或 ()

        # 1. 采样高斯噪声 ε ~ N(0, I)
        noise = jax.random.normal(noise_rng, actions.shape)

        # 2. 采样时间步 t ~ Beta(1.5, 1) * 0.999 + 0.001
        #    使用 Beta 分布让 t 略微偏向 1（噪声端），但保持在 (0.001, 1.0] 范围
        #    这是流匹配中的常见技巧，让模型在噪声端（推理起点）看到更多训练样本
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        # ========== RTC 训练模式（模拟推理延迟） ==========
        if self.rtc_simulated_delay is not None and self.rtc_simulated_delay > 0:
            # 在每个 batch 元素随机采样一个延迟 d ∈ [0, max_delay)
            delay_rng = jax.random.fold_in(time_rng, 1)
            max_delay = self.rtc_simulated_delay
            delay = jax.random.randint(
                delay_rng, batch_shape, minval=0, maxval=max_delay
            )  # (b,)

            # 构建逐位置的"已提交"掩码：位置 < delay → 已提交
            pos_idx = jnp.arange(self.action_horizon)[None, :]  # (1, H)
            delay_mask = pos_idx < delay[:, None]  # (b, H)

            # 逐位置的时间步：
            # - 已提交位置: time=0.0（干净动作/目标端）
            #   在流匹配中 t=0 对应目标分布（干净动作），t=1 对应噪声
            #   设 time=0 意味着 x_t = 0*noise + 1*actions = actions（纯干净动作）
            # - 未提交位置: time=原始采样值（正常训练）
            time_per_token = jnp.where(
                delay_mask, jnp.zeros_like(time[:, None]), time[:, None]
            )  # (b, H)

            # 构造逐位置的带噪样本：
            # 已提交位置: x_t = 0*noise + 1*actions = actions（干净）
            # 未提交位置: x_t = time*noise + (1-time)*actions（正常插值）
            time_expanded = time_per_token[..., None]  # (b, H, 1)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions  # 目标向量场（与位置无关）

            # ========== 前向传播 ==========
            # 1. 构建前缀（图像 + 语言）
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            # 2. 构建后缀（状态 + 噪声动作 + 逐位置时间）
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_per_token  # ← 传入逐位置时间 (b, H)
            )
            # 3. 拼接前缀和后缀的掩码
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1  # 0-based 位置索引

            # 4. LLM 前向传播
            # 前缀使用 VLM 专家（adarms_cond=None），后缀使用动作专家（adarms_cond=时间条件）
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            # 5. 输出投影：取后缀的最后 H 个位置 → 投影到动作空间
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # ========== 损失计算（只计算未提交位置） ==========
            per_token_loss = jnp.square(v_t - u_t)  # (b, H, action_dim)
            per_token_loss = jnp.mean(per_token_loss, axis=-1)  # (b, H) —— 对动作维度取均值
            # 已提交位置不参与损失计算（它们的 time=0 是 trivial 的）
            loss_mask = jnp.logical_not(delay_mask).astype(jnp.float32)  # (b, H)
            # 归一化：除以未提交位置数 + 小常数（避免除以 0）
            return jnp.sum(per_token_loss * loss_mask, axis=-1) / (
                jnp.sum(loss_mask, axis=-1) + 1e-8
            )  # (b,)

        # ========== 标准训练路径（无 RTC） ==========
        # 标量时间扩展到动作维度：(b,) → (b, 1, 1)
        time_expanded = time[..., None, None]
        # 线性插值构造带噪样本
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 目标向量场：从带噪样本指向干净样本的方向
        u_t = noise - actions

        # 一次前向传播处理前缀 + 后缀
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # LLM 前向传播：传入 [prefix_tokens, suffix_tokens] 这个「两条输入」的列表。
        # Gemma Module 按 configs 顺序依次处理每条输入（列表第 0 个 → VLM 专家，第 1 个 → 动作专家），
        # 返回同样长度的输出列表。adarms_cond=[None, adarms_cond] 按同一索引对应：
        #   前缀用 None（VLM 专家不需要时间条件），后缀用时间条件向量（动作专家需要）。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # 取后缀输出（最后 H 个 token）作为动作预测
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # MSE 损失：模型预测的向量场 vs 真实向量场
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # ===== RTC 参数 =====
        prev_chunk_left_over: at.Float[at.Array, "b ah ad"] | None = None,
        prev_chunk_left_over_len: int | at.Int[at.Array, ""] | None = None,
        inference_delay: int | at.Int[at.Array, ""] = 0,
        prefix_horizon: int | at.Int[at.Array, ""] | None = None,
        max_guidance_weight: float | at.Float[at.Array, ""] | None = None,
        trained_rtc_mode: bool = False,
    ) -> _model.Actions:
        """从噪声采样动作序列（去噪/推理过程）。

        这是 π₀ 的推理入口。它使用流匹配的逆过程从纯噪声逐步去噪，
        生成干净的动作序列。

        推理过程（无 RTC）：
        1. 初始化：x = noise（纯噪声，t=1）
        2. 循环 num_steps 次：
           a. 模型预测向量场 v_t = model(x, t)
           b. 更新：x = x + dt * v_t（沿向量场方向移动）
           c. 更新时间：t = t + dt
           （dt 为负值，如 -0.1，逐步向 t=0 推进）
        3. 返回 x（t≈0 时的干净动作）

        三种推理模式：
        1. 标准模式：无 RTC，一次性生成全部动作（while_loop）
        2. 训练时 RTC（trained_rtc_mode=True）：用固定前缀条件，无 VJP
        3. 推理时 RTC（prev_chunk_left_over 有值）：用 VJP 计算引导修正

        Args:
            rng: 随机数种子（用于初始化噪声）
            observation: 观测数据
            num_steps: 去噪步数（默认 10，越多越精细但越慢）
            noise: 可选，手动指定初始噪声
            prev_chunk_left_over: RTC 模式下，前一个 chunk 剩余的动作
                （形状 (b, H, action_dim)，可能是填充过的）
            prev_chunk_left_over_len: 前一个 chunk 剩余动作的实际长度
                （可能比 action_horizon 短）
            inference_delay: 推理延迟（已提交的前缀长度）
            prefix_horizon: RTC 引导的有效前缀长度
            max_guidance_weight: RTC 引导权重的最大值
            trained_rtc_mode: 是否使用训练时 RTC 推理（无 VJP，更快）

        Returns:
            形状 (b, H, action_dim) 的去噪动作序列
        """
        # 预处理观测（无数据增强，train=False）
        observation = _model.preprocess_observation(None, observation, train=False)

        # dt 为负值：从 t=1（噪声）向 t=0（目标）推进
        # 例如 num_steps=10 → dt = -0.1
        # 注意：这里遵循扩散文献中的常见约定（t=1 是噪声，t=0 是目标）
        # 这与 π₀ 论文相反，论文中 t=0 是噪声，t=1 是目标
        dt = -1.0 / num_steps

        batch_size = observation.state.shape[0]

        # 如果没提供初始噪声，采样标准高斯噪声
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ========== 第 1 步：预计算前缀的 KV Cache ==========
        # 前缀（图像 + 语言）在去噪过程中不变，可以预先计算并缓存
        # KV cache 让后续每次去噪迭代只需要传入后缀，大幅加速推理
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # 传入 [prefix_tokens, None]：只计算前缀，不计算后缀（后缀暂为空）
        # 返回 None（不需要前缀输出）和 KV cache
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # ========== RTC 模式路由 ==========
        if prev_chunk_left_over is not None:
            if trained_rtc_mode:
                # 训练时 RTC：直接将前几个位置固定为上一 chunk 的动作
                # 模型在训练中已学会基于已提交前缀生成后续动作
                return self._sample_actions_trained_rtc(
                    observation, noise, prefix_mask, kv_cache,
                    num_steps, dt, batch_size,
                    prev_chunk_left_over, inference_delay,
                )
            # 推理时 RTC（Kinetix VJP 方法）：用 VJP 计算引导修正
            # 引导模型生成的后续动作与上一 chunk 已提交的动作保持一致
            return self._sample_actions_rtc(
                observation, noise, prefix_mask, kv_cache,
                num_steps, dt, batch_size,
                prev_chunk_left_over, prev_chunk_left_over_len, inference_delay, prefix_horizon,
                max_guidance_weight,
            )

        # ========== 标准去噪循环（非 RTC） ==========
        # 使用 jax.lax.while_loop 实现，支持动态步数但编译为单个 XLA 图
        def step(carry):
            """单步去噪：x_{t+dt} = x_t + dt * v_t"""
            x_t, time = carry

            # 构建后缀（噪声动作 + 当前时间）
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # 后缀的注意力掩码（动作之间的因果注意力）
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            # 将前缀掩码扩展到后缀维度（后缀需要 attend 前缀）
            # (b, prefix_len) → (b, suffix_len, prefix_len)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])

            # 完整注意力掩码：后缀可以 attend 前缀 + 后缀（因果）
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )

            # 计算后缀 token 的绝对位置（在前缀之后继续数）
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # LLM 前向传播（复用前缀的 KV cache）
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],  # 前缀为 None（使用缓存），后缀传入新 token
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,      # ← 复用前缀的 KV cache
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None  # 前缀输出应为空（使用了缓存）

            # 取最后 H 个 token 投影为向量场
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 欧拉步：x_{t+dt} = x_t + dt * v_t
            # dt 为负值，所以是在减去噪声方向
            return x_t + dt * v_t, time + dt

        def cond(carry):
            """循环条件：当 time >= -dt/2 时继续（t≈0 时停止）"""
            x_t, time = carry
            # 使用 -dt/2 而非 0 作为阈值，避免浮点误差导致提前终止
            return time >= -dt / 2

        # 初始状态：x = noise (纯噪声, t=1)
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_actions_trained_rtc(
        self,
        observation: _model.Observation,
        noise: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        num_steps: int,
        dt: float,
        batch_size: int,
        prev_chunk_left_over: jax.Array,
        inference_delay: int | jax.Array,
    ) -> jax.Array:
        """训练时 RTC 推理模式：用固定前缀条件生成，无需 VJP。

        RTC（Real-Time Chunking）的目标：
        机器人在执行动作时，不需要等模型生成全部 H 步动作。可以将动作分成
        多个 chunk，先执行前几个动作，同时模型在后台继续生成后续动作。

        训练时 RTC 的工作原理：
        1. 在训练阶段就模拟了"部分动作已提交"的场景（见 compute_loss）
        2. 推理时直接将已提交的前几个位置固定为上一 chunk 的动作
        3. 这些位置的 time 设为 0（干净/目标端）
        4. 剩余位置正常去噪

        与推理时 RTC（VJP 方法）的对比：
        - 训练时 RTC：约 1× 推理成本（无 VJP 开销），但需要专门训练的模型
        - 推理时 RTC：约 2-3× 推理成本（每次迭代都要算 VJP），但可用普通模型

        为什么用 jax.lax.scan 而非 while_loop：
        scan 只编译一次循环体，避免展开 N 次导致 XLA 图爆炸。
        此处去噪步数 num_steps 是固定的，scan 比 while_loop 更高效。

        Args:
            observation: 观测数据
            noise: 初始噪声
            prefix_mask: 前缀的有效性掩码
            kv_cache: 预计算的前缀 KV cache
            num_steps: 去噪步数
            dt: 时间步长（负值）
            batch_size: batch 大小
            prev_chunk_left_over: 上一 chunk 剩余的动作（可能已填充到 H）
            inference_delay: 已提交的前缀长度（前几个位置固定不动）

        Returns:
            形状 (b, H, action_dim) 的去噪动作序列
        """
        # 确保 prev_chunk_left_over 填充到 action_horizon 长度
        # 真实剩余动作可能比 H 短（第一个 chunk 没有剩余）
        if prev_chunk_left_over.shape[1] < self.action_horizon:
            padded = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            padded = padded.at[:, :prev_chunk_left_over.shape[1], :].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        # 构建"已提交"掩码：前 inference_delay 个位置是已提交的
        # (1, H)，True = 已提交，False = 需要去噪
        committed_mask = jnp.arange(self.action_horizon)[None, :] < inference_delay  # (1, H)

        def step(carry, _):
            """单步去噪（训练时 RTC 版本）"""
            x_t, time = carry

            # ========== 核心：固定已提交位置 ==========
            # 将前 inference_delay 个位置强制设为上一 chunk 的动作
            # 这些位置不做去噪——它们已经是"正确答案"
            # committed_mask (1, H) 经 [:, :, None] 广播成 (1, H, 1)，
            # 与 prev_chunk_left_over / x_t 的 (b, H, ad) 按元素广播（batch 维也广播）。
            x_t = jnp.where(committed_mask[:, :, None], prev_chunk_left_over, x_t)

            # 逐位置时间：已提交位置 time=0（干净/目标），其余位置用当前 time
            # 这复制了训练时 RTC 的设置（见 compute_loss 中的 delay_mask）
            time_per_token = jnp.where(
                committed_mask, jnp.zeros((batch_size, self.action_horizon)),
                jnp.broadcast_to(time, (batch_size, self.action_horizon))
            )  # (b, H)

            # 前向传播（带逐位置时间）
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_per_token
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 更新：只更新未提交位置
            x_t_new = x_t + dt * v_t
            # 再次将已提交位置固定为上一 chunk 的动作（防止漂移）
            x_t_new = jnp.where(committed_mask[:, :, None], prev_chunk_left_over, x_t_new)

            return (x_t_new, time + dt), None

        # jax.lax.scan: 循环 num_steps 次，每次执行 step
        # 第一个返回值 (x_0, _) 是最终的 x_t 和 time
        (x_0, _), _ = jax.lax.scan(step, (noise, jnp.array(1.0)), length=num_steps)
        return x_0

    def _sample_actions_rtc(
        self,
        observation: _model.Observation,
        noise: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        num_steps: int,
        dt: float,
        batch_size: int,
        prev_chunk_left_over: jax.Array,
        prev_chunk_left_over_len: int | jax.Array | None,
        inference_delay: int | jax.Array,
        prefix_horizon: int | jax.Array | None,
        max_guidance_weight: float | jax.Array | None,
    ) -> jax.Array:
        """推理时 RTC：使用 Kinetix VJP 方法进行前缀引导的动作采样。

        ╔══════════════════════════════════════════════════════════════════╗
        ║              推理时 RTC 核心思想（Kinetix 方法）                  ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║                                                                  ║
        ║  问题：模型生成的动作序列需要与上一 chunk 已提交的前几个动作       ║
        ║  一致（连贯性），但普通去噪过程不知道上一 chunk 是什么。          ║
        ║                                                                  ║
        ║  解决方案：在每次去噪迭代中，用 VJP（Vector-Jacobian Product）    ║
        ║  计算"如何改变当前 x_t 能让预测的 x_0 更接近上一 chunk 的动作"。  ║
        ║                                                                  ║
        ║  数学推导：                                                      ║
        ║  1. 定义 error = (prev_chunk - x_0) * weights                   ║
        ║     其中 x_0 = denoiser(x_t) = x_t - t * v_t                    ║
        ║  2. 用 VJP 计算 pinv = (∂x_0/∂x_t)^T · error                   ║
        ║     这相当于伪逆（pseudoinverse）校正方向                         ║
        ║  3. 修正向量场：v_t' = v_t - guidance_weight * pinv             ║
        ║     使去噪后的动作向上一 chunk 的动作靠拢                         ║
        ║                                                                  ║
        ╚══════════════════════════════════════════════════════════════════╝

        为什么用 jax.lax.scan 而非 Python for 循环：
        JIT 编译会展开 Python for 循环 N 次（这里 N=num_steps）。如果每次迭代
        都包含 VJP（本身就很大），展开 N 次会导致 XLA 图爆炸——编译极慢且
        可能 OOM。scan 只编译循环体一次，编译后复用 N 次，大幅减少编译开销。

        时间约定差异（重要！）：
        - Kinetix: t=0 是噪声，t=1 是目标，v_t = actions - noise，dt > 0
        - openpi:  t=1 是噪声，t=0 是目标，v_t = noise - actions，dt < 0

        因为 openpi 的 v_t 符号与 Kinetix 相反，修正必须是减法而非加法：
          v_t_corrected = v_t - guidance_weight * pinv_correction  （减法！）
        这等价于 Kinetix 的 v_t + guidance * correction（考虑符号翻转后）。

        Args:
            observation: 观测数据
            noise: 初始噪声
            prefix_mask: 前缀的有效性掩码
            kv_cache: 预计算的前缀 KV cache
            num_steps: 去噪步数
            dt: 时间步长（负值）
            batch_size: batch 大小
            prev_chunk_left_over: 上一 chunk 剩余的动作
            prev_chunk_left_over_len: 上一 chunk 剩余动作的实际长度
            inference_delay: 推理延迟（已提交的前缀长度）
            prefix_horizon: RTC 引导覆盖的前缀长度
            max_guidance_weight: 引导权重的最大值（防止修正过大）

        Returns:
            形状 (b, H, action_dim) 的去噪动作序列
        """
        # ========== RTC 参数初始化 ==========
        # 从 self 读取（module_jit 时的 Python 常量，类似 self.pi05）
        prefix_attention_schedule = self.rtc_prefix_schedule  # "exp" 指数衰减
        if max_guidance_weight is None:
            max_guidance_weight = self.rtc_max_guidance_weight  # 默认 10.0
        if prev_chunk_left_over_len is None:
            prev_chunk_left_over_len = prev_chunk_left_over.shape[1]

        # ========== 填充上一 chunk 的动作 ==========
        # 如果上一 chunk 的剩余动作比 H 短，填充 0 到 H 长度
        # JAX 要求固定形状的输入，不能动态变化
        if prev_chunk_left_over.shape[1] < self.action_horizon:
            padded = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            padded = padded.at[:, :prev_chunk_left_over.shape[1], :].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        # ========== 计算有效的引导范围 ==========
        # 限制引导范围不超过：
        # 1. prefix_horizon（用户指定的引导长度）
        # 2. prev_chunk_left_over_len（上一 chunk 的实际剩余长度）
        #    填充的 0 不应该参与引导计算
        # 3. action_horizon（模型的最大预测长度）
        if prefix_horizon is None:
            prefix_horizon = self.action_horizon
        effective_horizon = jnp.minimum(prefix_horizon, prev_chunk_left_over_len)
        effective_horizon = jnp.minimum(effective_horizon, self.action_horizon)

        # ========== 计算前缀权重（循环外计算一次，适用于所有步） ==========
        # get_prefix_weights 生成一个随时间衰减的权重向量
        # "exp" 策略：指数衰减，越靠后的位置对前缀依赖性越弱
        # 这是合理的：距离已提交前缀越远的动作，越不应该被强约束
        # 形状: (1, T, 1) —— batch 和 action_dim 广播
        weights = get_prefix_weights(
            inference_delay, effective_horizon, self.action_horizon, prefix_attention_schedule
        )[None, :, None]  # (1, T, 1)

        def step(carry, _):
            """单步去噪（VJP RTC 版本）—— 这是整个 RTC 实现中最关键的函数"""
            x_t, time = carry
            expanded_time = jnp.broadcast_to(time, (batch_size,))

            # ========== 定义去噪函数（VJP 的目标函数） ==========
            # denoiser(x_t) 接受当前带噪样本、运行完整前向传播、返回预测的 x_0
            # jax.vjp 将自动计算 ∂x_0/∂x_t 的完整导数（通过整个模型的反向传播）
            def denoiser(x_t_arg):
                """完整的模型前向传播，返回预测的干净样本 x_0 和向量场 v_t"""
                # 构建后缀
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t_arg, expanded_time
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

                # 去噪预测：x_0 = x_t - t * v_t（openpi 约定）
                # 推导：由 x_t = t*noise + (1-t)*x_0, v_t = noise - x_0
                #      → x_t - t*v_t = t*noise + (1-t)*x_0 - t*(noise - x_0)
                #      = t*noise + x_0 - t*x_0 - t*noise + t*x_0 = x_0 ✓
                x_0 = x_t_arg - time * v_t
                return x_0, v_t  # has_aux=True → v_t 作为辅助输出

            # ========== Kinetix 核心：VJP 计算引导修正 ==========
            # jax.vjp(denoiser, x_t) 返回：
            #   - primals: denoiser 的原始输出 (x_0, v_t)
            #   - vjp_fn:  一个函数，接受 ∂loss/∂x_0 的梯度，返回 ∂loss/∂x_t
            #              即实现 (∂x_0/∂x_t)^T 的向量-雅可比乘积
            # 这需要对整个 LLM 做反向传播，所以推理时 RTC 比训练时 RTC 慢 2-3 倍
            x_0, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)

            # ========== 计算 error 和伪逆校正 ==========
            # error: 预测的 x_0 偏离上一 chunk 动作的程度
            # prev_chunk_left_over - x_0: 预测值离目标有多远
            # * weights: 不同位置的重要性不同（靠近前缀的更重要）
            error = (prev_chunk_left_over - x_0) * weights

            # pinv_correction = (∂x_0/∂x_t)^T · error
            # 这告诉我们：沿着什么方向改变 x_t 可以最有效地减小 error
            # 之所以叫"伪逆"（pseudoinverse），是因为 (∂x_0/∂x_t) 通常不是方阵
            # vjp_fn 返回的是一个 (∂x_0/∂x_t)^T 与梯度的乘积
            pinv_correction = vjp_fn(error)[0]

            # ========== 计算引导权重（自适应缩放） ==========
            # 转换为 Kinetix 的时间约定：
            # tau = 1 - time: Kinetix 中 tau 从 0→1（噪声→目标）
            # one_minus_tau = time: Kinetix 中 1-tau 从 1→0
            tau = 1.0 - time
            one_minus_tau = time  # = 1 - tau（在 Kinetix 约定中）

            # inv_r² = ((1-tau)² + tau²) / ((1-tau)² + 1e-8)   ← 代码实现，分母加了防除零常数
            # 这是流匹配时间步的自适应缩放因子（tau = 1 - time），行为如下：
            #   time→0（去噪末端，tau→1）：分母 (1-tau)²→0，inv_r² 急剧放大；
            #   time→1（去噪起点，tau→0）：inv_r² ≈ 1，缩放平缓。
            # 它与下方 c 系数相乘共同决定引导强度 guidance_weight，最后裁切到上限。
            inv_r2 = (one_minus_tau**2 + tau**2) / (one_minus_tau**2 + 1e-8)

            # c = (1-tau)/tau：KL 散度相关的系数
            # 当 tau 很小时（去噪早期）c 很大 → 引导强
            # 当 tau 变大时（去噪后期）c 变小 → 引导弱
            c = jnp.where(tau > 1e-8, one_minus_tau / tau, max_guidance_weight)

            # guidance_weight = c * inv_r²，裁切到 [0, max_guidance_weight]
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

            # ========== 修正向量场 ==========
            # 减法而非加法！因为 openpi 的 v_t 方向与 Kinetix 相反
            # Kinetix: v_t = actions - noise → 修正 v_t' = v_t + guidance * correction
            # openpi:  v_t = noise - actions → 修正 v_t' = v_t - guidance * correction
            # 两种写法等价（考虑符号翻转和 error 定义后）
            v_t = v_t - guidance_weight * pinv_correction

            # ========== 欧拉步 ==========
            x_t = x_t + dt * v_t
            time = time + dt

            return (x_t, time), None

        # scan 执行 num_steps 次去噪迭代
        # 每次迭代包含：完整前向传播 + VJP 反向传播（成本约 2-3 倍普通前向）
        (x_0, _), _ = jax.lax.scan(step, (noise, jnp.array(1.0)), length=num_steps)
        return x_0
