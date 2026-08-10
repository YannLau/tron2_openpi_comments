# 中文 | [English](README.md)

# TRON2 OpenPI

`tron2_openpi` 是 OpenPI 项目的 TRON2 部署派生版本。保留 OpenPI 的策略服务和 pi0/pi0.5 模型栈，额外增加了 TRON2 策略变换、部署配置模板和真机客户端示例。

本仓库需与兄弟仓库 `tron2_env` 配合使用。

## 本仓库提供

- 通过 `scripts/serve_policy.py` 提供 pi0.5 策略服务
- `src/openpi/policies/tron2_policy.py` 中的 TRON2 策略输入输出变换
- `examples/tron2/` 中的 TRON2 机器人客户端
- `configs/deploy/` 中的公开部署模板
- 基于 YAML 的 TRON2 任务训练
- 可选的 Bridge 观测模式和 RealSense 相机模式
- OpenPI 客户端包（`packages/openpi-client/`）

## 未包含的内容

- 模型权重和检查点目录
- 训练/评估数据集
- 私有 `.local.yaml` 部署文件
- 安全认证

## 示例任务

| 任务 | 用户指南 | 模型权重 | 部署配置 |
|------|----------|----------|----------|
| Candy | [用户指南](https://cwjgfm21di.feishu.cn/wiki/NA5Rw1dWPiu6dwkFAfTcnaFLnQf) | [HuggingFace](https://huggingface.co/limx-tron2-dev/tron2-openpi-models) | `candy_server.yaml`, `candy_client.yaml` |
| Cloth | [中文指南](https://cwjgfm21di.feishu.cn/wiki/Bcw8wthgpiLrVWkHXk0cBfLOnnc) | [HuggingFace](https://huggingface.co/limx-tron2-dev/tron2-openpi-models) | `cloth_server.yaml`, `cloth_client.yaml` |

## 安装

环境要求与配置命令详见 [INSTALL.md](INSTALL.md) 和 [INSTALL_CN.md](INSTALL_CN.md)。

## 部署

### 运行策略服务

启动策略服务器：

```bash
uv run scripts/serve_policy.py --profile configs/deploy/candy_server.yaml
```

运行客户端：

```bash
uv run examples/tron2/pi_client.py --profile configs/deploy/my_task_client.local.yaml
```

## 训练新的 TRON2 任务

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
uv run scripts/compute_norm_stats.py --task-config configs/train/tron2_tasks/my_task.yaml
uv run scripts/train_tron2_task.py --task-config configs/train/tron2_tasks/my_task.yaml
```

## 相关链接

- [tron2_env](https://github.com/limxdynamics/tron2_env) — TRON2 运行环境
- [tron2-robot-description](https://github.com/limxdynamics/tron2-robot-description) — 机器人模型
