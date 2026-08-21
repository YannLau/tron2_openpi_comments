"""pi0 / pi05 模型的配置类。

`Pi0Config` 是 openpi 中最常用的模型配置：它继承了 `model.py` 里的 `BaseModelConfig` 基类，
把 pi0 系列模型的“默认超参数”和“如何实例化模型 / 如何描述输入形状 / 如何做参数冻结”这几件事集中在一处。

整个链路是这样协作的：

    Pi0Config（本文件，纯配置，frozen dataclass）
        ├── .create(rng)  ──────────────▶ 构造 Pi0 模型（见 openpi/models/pi0.py）
        ├── .inputs_spec() ─────────────▶ 给 XLA 做静态形状推导 / jit 预热
        ├── .load() / .load_pytorch() ──▶ 继承自基类，装载 checkpoint 权重
        └── .get_freeze_filter() ───────▶ 微调（LoRA）时决定哪些参数冻结、哪些可训练

理解 pi0 架构的两块组成部分，很多字段就一目了然了：
1. **VLM 主干（PaliGemma）**：视觉编码器（SigLIP）+ 语言模型（Gemma）。负责“看图 + 读语言指令”，
   把观测信息编码成 token 序列。由 `paligemma_variant` 选择规模。
2. **动作专家（Action Expert）**：一个更小的 Gemma 网络。在 VLM 输出末尾追加“动作 token”，
   通过流匹配 / 扩散（pi0）或自回归（pi0_fast）方式生成动作轨迹。由 `action_expert_variant` 选择规模。

因此 `*_variant` 选的是“Gemma 的语言模型规模”，比如 "gemma_2b"（约 2B 参数）/"gemma_300m"（约 300M 参数）。
"""

import dataclasses  # frozen dataclass：配置对象不可变，可安全共享
from typing import TYPE_CHECKING  # 类型检查专用导入：只用于 type checker，避免运行期循环导入

import flax.nnx as nnx  # Flax 神经网络库（NNX API），这里用到 nnx.Rngs / nnx.filterlib.Filter
import jax  # JAX 数组框架
import jax.numpy as jnp  # JAX 版 numpy
from typing_extensions import override  # 显式标注“此方法覆盖父类方法”，增强可读性

# 以 _model / _gemma 别名导入，避免与模块名冲突，同时提示读者它们来自哪里。
from openpi.models import model as _model  # 基类 BaseModelConfig / Observation / Actions 等
import openpi.models.gemma as _gemma  # Gemma 语言模型的配置（Variant 类型、get_config 等）
from openpi.shared import array_typing as at  # 数组类型别名与运行时类型检查
import openpi.shared.nnx_utils as nnx_utils  # NNX 工具，这里用到 PathRegex（按正则匹配参数路径的过滤器）

# TYPE_CHECKING 为 True 时（仅 type checker / IDE 阶段）才导入 Pi0。
# 运行时不会执行该导入，因此即使 pi0.py 反过来 import 本文件，也不会形成循环导入错误。
if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """pi0 / pi05 的模型配置类。

    继承 `BaseModelConfig`（见 model.py），因此拥有 action_dim / action_horizon / max_token_len
    三个共享字段，以及 load() / load_pytorch() / fake_obs() / fake_act() 等通用方法。
    这里再补上 pi0 特有的超参数，并用 `@override` 覆写 create() / inputs_spec() / model_type。
    """

    # ---- 精度 / 网络规模 ----

    # 模型权重与激活的主精度。bfloat16 是训练大模型的标准选择：
    # 相比 float32 省一半显存，数值范围又与 float32 相同，足够稳定。
    dtype: str = "bfloat16"
    # VLM 主干（PaliGemma = SigLIP 视觉编码器 + Gemma 语言模型）的规模。
    # 可选值见 gemma.Variant："gemma_2b" / "gemma_2b_lora" / "gemma_300m" / "gemma_300m_lora" / "dummy"。
    #   - 带 "_lora" 后缀：使用 LoRA 微调（冻结主干，只训练少量低秩适配矩阵，见 get_freeze_filter）；
    #   - "dummy"：极小的哑模型，用于跑通代码 / 单元测试。
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # 动作专家（Action Expert）的规模。默认 gemma_300m（约 300M 参数），比主干小很多——
    # 动作专家只需把“已编码好的观测”映射到动作，负担较轻，小模型更快更省显存。
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # ---- 输入 / 输出形状（pi0 系列默认值）----

    # 动作空间的维度（每个时间步动作向量的长度）。默认 32：
    # 常见的机械臂是 7 个关节角 + 1 个夹爪 = 8 维，openpi 用 32 维支持更多自由度（如双机械臂）。
    action_dim: int = 32
    # 动作轨迹长度：一次预测未来多少步动作。默认 50 步。
    # （假设控制频率 50Hz，50 步 ≈ 1 秒的动作；配合“滚动执行 + 每次重新预测”实现平滑控制。）
    action_horizon: int = 50
    # tokenized prompt 的最大长度。这里写 None 并标注 type: ignore（故意与基类类型不符），
    # 在 __post_init__ 中根据 pi05 自动补成实际值：pi05 用 200，pi0 用 48。
    max_token_len: int = None  # type: ignore

    # ---- 架构版本开关 ----

    # Pi0.5 与 Pi0 的两点区别（在 pi0.py 的 Pi0 类中实现）：
    #   1. 机器人状态（state）作为“离散语言 token”拼进 prompt，而不是作为连续输入追加在 suffix 里；
    #      —— 即 state 先被量化成 token 交给语言模型，让 VLM 可以像“读指令”一样读状态。
    #   2. 动作专家用 adaRMSNorm 注入流匹配的时间步（timestep）条件。
    # 置 True 即启用 pi05 架构，此时 model_type 返回 ModelType.PI05。
    pi05: bool = False
    # 状态是否以离散 token 输入。注意：模型本身不直接读这个字段，
    # 它由数据变换侧（transforms.py 的 TokenizePrompt，见 discrete_state_input 参数）读取，
    # 用于决定是否把 state 分词后拼进 prompt。
    # 默认 None 表示“跟随 pi05”：pi05=True 时自动为 True。
    discrete_state_input: bool = None  # type: ignore

    # ---- 训练时 RTC（Real-Time Control，实时控制）模拟 ----

    # 在训练阶段模拟推理延迟，见下方 docstring。默认 None = 不启用。
    rtc_training_simulated_delay: int | None = None

    # When set to a positive int, each training sample randomly masks the first d
    # positions (d ~ [0, rtc_training_simulated_delay)) as "already committed" by
    # setting their time to 1.0 (pure noise) and zeroing their loss contribution.
    # This teaches the model to denoise conditioned on a frozen prefix.
    #
    # （中文释义）真实实时控制中，模型输出动作需要计算时间，因此发出动作时“前几步其实已经执行了”。
    # 开启本选项后，每个训练样本会随机选取前 d 步（d 在 [0, delay) 内随机）当作“已经承诺执行”：
    # 把它们的噪声时间 t 设为 1.0（即纯噪声）并把它们的损失权重清零。
    # 这样训练出来的模型学会“在已执行轨迹冻结的条件下继续去噪”，与推理时真正的延迟场景对齐。

    def __post_init__(self):
        """dataclass 初始化后的钩子：把延迟到 __post_init__ 才确定的默认值补全。

        用 object.__setattr__ 赋值是因为 frozen=True 的 dataclass 不允许直接写属性，
        这个技巧是 frozen dataclass 的官方推荐写法。
        """
        # max_token_len：pi05 需要把 state 也 token 化拼进 prompt，序列更长，故用 200；pi0 只需 48。
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        # discrete_state_input 默认跟随 pi05。
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override  # 覆盖基类的抽象属性 model_type（见 BaseModelConfig）
    def model_type(self) -> _model.ModelType:
        """返回模型类型：pi05 架构对应 ModelType.PI05，否则 ModelType.PI0。"""
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        """创建（随机初始化）一个 pi0 模型实例。

        这是 BaseModelConfig.create 的实现。函数内 import 是为了避免循环导入
        （pi0.py 顶层会 import 本模块，因此本模块只能在函数体内延迟导入 Pi0）。
        """
        from openpi.models.pi0 import Pi0

        # nnx.Rngs 是 NNX 的随机源容器：把同一个 key 交给模型，模型内部为各子模块派生独立随机 key。
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """返回模型输入 / 输出的“形状规格”，供 XLA 静态编译与 fake_obs()/fake_act() 使用。

        返回的每个数组都是 jax.ShapeDtypeStruct——只有 shape 和 dtype、没有真实数据，
        因此可以零成本地描述“模型要吃进什么形状的输入”。

        - 图片规格：与 _model.IMAGE_KEYS 对齐，三路相机（base / left_wrist / right_wrist），
          布局 NHWC，分辨率 _model.IMAGE_RESOLUTION（224×224），3 通道，float32。
        - 图片 mask：每个 batch 元素一个 bool，标识图像是否有效。
        - state：形状 [batch_size, action_dim]，float32（注意：state 维度与动作维度一致）。
        - prompt：int32 token 序列，长度固定为 max_token_len（不足部分由 dataloader padding）。
        - 动作：形状 [batch_size, action_horizon, action_dim]，float32。
        """
        # 单路相机的形状规格：[*b, 224, 224, 3]，即 NHWC 的 224×224 彩色图。
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 单路相机 mask 的形状规格：每个样本一个 bool。
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # 构造 Observation 规格对象。注意这里塞进去的是 ShapeDtypeStruct（无真实数组），
            # 会触发 jaxtyping 的类型检查报错，因此临时禁用类型检查。
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # state 的形状：[batch_size, action_dim]。
                # 之所以与 action_dim 一致，是因为模型把“当前状态 + 目标”作为条件，状态本身常由动作维描述。
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # prompt 与 prompt mask：长度固定为 max_token_len（pi0 为 48，pi05 为 200）。
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        # 动作规格：轨迹形状 [batch_size, action_horizon, action_dim]。
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.
        （根据配置返回“冻结过滤器”，用于决定微调（LoRA）时哪些参数冻结、哪些可训练。

        NNX 的 Filter 是一个可调用对象：给定参数路径 + 值，返回 True 表示“该参数被过滤器命中”。
        训练脚本（training/config.py）拿到这个 filter 后，把命中的参数标记为不可训练。

        参数路径命名约定（模型内部结构）：
          - 含 "llm" 的路径 → VLM 语言模型（Gemma）部分；
          - 含 "llm" 且以 "_1" 结尾 → 动作专家（Action Expert），它是第二个 Gemma 实例，故后缀 _1；
          - 含 "lora" 的路径 → LoRA 低秩适配矩阵参数。

        逻辑分两种场景：
          a) 完全微调（两个 variant 都不含 "lora"）→ 什么都不冻结（返回 nnx.Nothing，全部可训练）；
          b) 至少一侧用了 LoRA → 冻结基座主干（Gemma 大模型）的原始权重，只让 LoRA 矩阵可训练。
        """
        filters = []  # 收集所有“需要冻结”的过滤器
        has_lora = False  # 是否启用了 LoRA（决定最终是否要排除 lora 参数）

        # 匹配所有 VLM 语言模型参数（包括主干 llm 与动作专家 llm_1）。
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        # 匹配动作专家参数（路径含 "llm" 且以 "_1" 结尾——它是第二个 Gemma 实例）。
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            # 主干用 LoRA：冻结整块 Gemma 参数（主干 llm + 动作专家 llm_1 一起）。
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 只有主干用了 LoRA、动作专家没用时，动作专家应该保持“全参可训练”用于微调，
                # 所以从冻结清单里排除掉它（nnx.Not 表示取反）。
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            # 只有动作专家用 LoRA：只冻结动作专家（llm_1）的原始权重，主干保持冻结前状态。
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 既然启用了 LoRA，就必须保证 LoRA 矩阵本身不被冻结（否则就什么都没得训练了）。
            # 因此在冻结清单里排除所有 "lora" 参数。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            # 没有任何 LoRA：全部参数可训练，返回 nnx.Nothing（NNX 的“什么都不匹配”空过滤器）。
            return nnx.Nothing
        # 所有条件取交集（全部满足才算命中），得到最终的冻结过滤器。
        return nnx.All(*filters)
