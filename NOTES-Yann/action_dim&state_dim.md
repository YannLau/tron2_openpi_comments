## 结论先说

你的直觉基本对，但有一处措辞要修正：**不是"32 个噪声"，而是"每个时间步一个 32 维的噪声向量"**。噪声张量形状是 `(B, H, 32)`，其中 `H = action_horizon = 50`。模型对这 32 维做的是**联合去噪**（一个 32 维向量场），不是 32 条独立轨迹。去噪完成后确实只取真实动作对应的那 16 维。

真正的关系是：**32 维 = 真实的 16 维 + 尾部补的 16 个零**。下面把每一环串起来。

## 16 → 32 是在哪里发生的

真实的 16 维动作（左臂 7 关节 + 左夹爪 1 + 右臂 7 关节 + 右夹爪 1）在进入模型前被补零拉长到 32 维，由 [transforms.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/transforms.py:328) 的 `PadStatesAndActions` 完成，底层是 [pad_to_dim](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/transforms.py:423)，沿最后一维在**末尾追加 0**：

| 维度索引 | 内容 | 训练时的值 |
|---|---|---|
| 0–6 | 左臂 7 关节 | 真实动作 |
| 7 | 左夹爪 | 真实动作 |
| 8–14 | 右臂 7 关节 | 真实动作 |
| 15 | 右夹爪 | 真实动作 |
| 16–31 | 补位 | 恒为 0 |

数据管道的顺序是：repack → `Tron2Inputs`/`DeltaActions` → `Normalize`(16 维统计量) → `TokenizePrompt` → `PadStatesAndActions`(16→32)。所以**补进去的 0 是在归一化空间里的 0**，也就是"这个维度的均值"。[config.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/training/config.py:253) 里把这个变换接在 `Pi0Config.action_dim` 上。

补零同时作用于 `state` 和 `actions`，所以 `state_proj` 和 `action_in_proj` 的输入宽度都是 32（见 [pi0.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/models/pi0.py:247)）。

## 为什么是 32 而不是直接用 16

因为 `Pi0Config.action_dim` 默认就是 32（[pi0_config.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/models/pi0_config.py:70)），而 TRON2 的训练配置构造 `Pi0Config(pi05=True, ...)` 时**没有覆盖它**，所以沿用了 32。这不是 TRON2 特有的设计，DROID 的 pi0.5 微调配置同样是 `action_dim=32`（[config.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/training/config.py:1236)）。

关键原因：`pi05_base` / DROID 的预训练权重里，`action_in_proj` 的 kernel 形状是 `(32, width)`、`action_out_proj` 是 `(width, 32)`。如果你把 `action_dim` 改成 16，这两层的形状就对不上，checkpoint 加载会失败。想让形状匹配又复用预训练权重，就只能保留 32。

## 训练时这 32 维发生了什么

看 [compute_loss](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/models/pi0.py:519)：

```python
noise = jax.random.normal(noise_rng, actions.shape)   # (B, H, 32)
x_t   = t * noise + (1 - t) * actions                  # (B, H, 32)
u_t   = noise - actions                                 # 目标向量场
v_t   = self.action_out_proj(suffix_out[:, -H:])        # (B, H, 32)
loss  = mean(square(v_t - u_t), axis=-1)                # 对 32 维取均值
```

对前 16 维：`u_t = noise - 真实动作`，模型学的是"从噪声到真实动作"的向量场。
对后 16 维：`actions = 0`，于是 `u_t = noise - 0 = noise`，意味着 `x_0` 必须收敛到 0。**填充维不是被忽略，而是被专门训练成"输出去噪到 0"**。

有个副作用值得知道：`mean(..., axis=-1)` 是在全部 32 维上平均的，所以真实 16 维的梯度被稀释了大约一半。功能上无害，但如果从零训练，直接设 `action_dim=16` 会更干净。

另外注意 `action_dim` **不影响序列长度**：suffix 始终是 `H=50` 个 token，只是每个 token 的特征从 32 维投影出来而不是 16 维。所以 32 vs 16 对计算量的影响很小，只在投影层和逐元素运算上。

## 推理时怎么回到 16 维

`sample_actions` 从纯噪声 `(B, H, 32)` 出发，迭代 10 步（或 RTC 里的 N 步）积分向量场，输出仍是 `(B, H, 32)`（[pi0.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/models/pi0.py:680)）。之后输出侧两步把它变回 16 维：

1. `Unnormalize`：norm_stats 只有 16 维，`pad_to_dim` 把 `mean` 补 0、`std` 补 1，于是第 16–31 维是恒等映射，原样通过（它们此时已经接近 0）。
2. `Tron2Outputs` 用 `[..., :output_dim]` 截断到 16（[tron2_policy.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/src/openpi/policies/tron2_policy.py:107)），`output_dim` 来自 `state_dim`。

所以输出给真机的就是干净的 16 维。客户端里也能看到这个约定的印证：`action[:7]` 和 `action[8:15]` 拼出双臂关节、索引 7 和 15 是夹爪（[pi_client_rtc.py](/home/punk/yann_repo/openpi-series/yann-comments/tron2_openpi/examples/tron2/pi_client_rtc.py:1294)）。

## 一句话总结

32 是"模型的动作向量宽度"，16 是"真机的自由度"。补零把 16 垫到 32，flow matching 在这 32 维上联合去噪（后 16 维被训练成收敛到 0），最后截断前 16 维还原真机动作。保留 32 主要是为了兼容 `pi05_base` 预训练权重的投影层形状；代价是后 16 维白白占用了一半的 loss 权重和动作表示容量。