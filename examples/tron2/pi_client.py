"""Run a synchronous TRON2 real-robot OpenPI client.

=============================================================================
概述 / Overview
=============================================================================
这个脚本是 TRON2 真机部署的 **主控客户端**。它的核心职责是：

1. 连接到 TRON2 机器人环境（真机或仿真）
2. 通过 WebSocket 连接到远程策略服务器（OpenPI policy server）
3. 循环执行"观察 → 推理 → 执行"的控制闭环：
   - 从机器人获取当前状态（关节角度、图像等）
   - 将状态发送给策略服务器，获取动作预测
   - 在机器人上执行预测的动作
   - 按固定频率（fps）重复上述过程

简单来说，它是机器人"大脑"（策略模型）和"身体"（机械臂）之间的桥梁。

=============================================================================
依赖关系 / Dependencies
=============================================================================
- _external_tron2_env.py : TRON2 环境的路径配置辅助
- deploy_config.py      : 部署配置加载、观测格式化、推理计时、WebSocket 连接参数
- openpi_client          : OpenPI 的 WebSocket 客户端库（策略通信）
- tron2_env              : TRON2 机器人环境封装（仿真/真机统一接口）
- numpy                  : 数值计算，用于动作数组的拼接和保存
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# ---------------------------------------------------------------------------
# OpenPI WebSocket 客户端 —— 负责和远程策略服务器通信
# ---------------------------------------------------------------------------
from openpi_client import websocket_client_policy

# ---------------------------------------------------------------------------
# 本地辅助模块（examples/tron2/ 目录下）
# ---------------------------------------------------------------------------
from _external_tron2_env import ensure_external_tron2_env_on_path
from deploy_config import build_env_config       # 从 YAML 配置构建环境参数
from deploy_config import bool_value             # 安全解析布尔型配置项
from deploy_config import format_obs             # 将原始观测格式化为策略模型的输入
from deploy_config import infer_with_timing      # 执行一次推理并记录各阶段耗时
from deploy_config import load_deploy_config     # 加载部署 YAML 配置文件
from deploy_config import policy_host            # 从配置中提取策略服务器主机名
from deploy_config import policy_port            # 从配置中提取策略服务器端口
from deploy_config import positive_int_or_none   # 安全解析"正整数或 None"型配置
from deploy_config import PromptController       # 运行时动态修改 prompt 的控制器
from deploy_config import record_paths           # 生成录制数据的输出文件路径
from deploy_config import section                # 从配置字典中提取子配置段落
from _external_tron2_env import ensure_external_tron2_env_on_path
from deploy_config import PromptController
from deploy_config import bool_value
from deploy_config import build_env_config
from deploy_config import format_obs
from deploy_config import infer_with_timing
from deploy_config import load_deploy_config
from deploy_config import policy_host
from deploy_config import policy_port
from deploy_config import positive_int_or_none
from deploy_config import record_paths
from deploy_config import section
from deploy_config import select_profile_path
import numpy as np
from openpi_client import websocket_client_policy

# ---------------------------------------------------------------------------
# 在执行任何 TRON2 相关导入之前，先确保外部 TRON2 环境包的路径已加入 sys.path。
# 这样 Python 才能找到 tron2_env 模块（它可能安装在非标准位置）。
# ---------------------------------------------------------------------------
ensure_external_tron2_env_on_path()

from tron2_env import Tron2Env


# ===========================================================================
# 命令行参数解析
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含解析后参数的对象，有 deploy_config 和 prompt 两个属性。
    """
    parser = argparse.ArgumentParser(description="Run TRON2 real-robot policy client.")

    # --deploy-config: 指向部署配置 YAML 文件的路径
    # 这个 YAML 文件包含了所有部署参数：机器人连接方式、策略服务器地址、
    # 控制频率、录制选项等。如果不传则使用代码中的默认值。
    parser.add_argument("--profile", type=str, default=None, help="Path to client deployment profile YAML.")
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Deprecated alias for --profile.",
    )
    # --prompt: 任务指令文本，会和每次观测一起发送给策略模型
    # 例如 "pick up the red block" 或 "open the drawer"
    # 如果同时设置了 YAML 中的 client.prompt，命令行参数优先级更高
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt sent with each observation. Overrides client.prompt in YAML.",
    )

    return parser.parse_args()


# ===========================================================================
# 数据录制 —— 保存动作和状态到 CSV 文件
# ===========================================================================
def _save_records(
    config_profile: dict,
    actions: list[np.ndarray],
    states: list[np.ndarray],
) -> None:
    """将执行过程中收集的动作和状态数据保存到 CSV 文件。

    这在调试、回放和分析时非常有用。保存的文件路径由部署配置中的
    action_output_path 和 state_output_path 字段决定。

    Args:
        config_profile: 完整的部署配置字典（从 YAML 加载）。
        actions: 动作数组的列表，每个元素是一个 (N_steps, action_dim) 的数组，
                 其中 N_steps 是策略输出的动作分步数（action chunking）。
        states: 状态数组的列表，每个元素是一个 (state_dim,) 的一维数组。
    """
    # 如果没有收集到任何数据，直接返回（例如 max_steps=0 的情况）
    if not actions or not states:
        return

    # 根据配置生成输出文件路径
    action_path, state_path = record_paths(
        config_profile,
        action_key="action_output_path",
        state_key="state_output_path",
        action_suffix="action_data",
        state_suffix="state_data",
    )

    # 将所有 step 的动作纵向堆叠成一个大的二维数组，保存为 CSV
    # 例如：如果有 100 个 step，每个 step 输出 10 个分步动作，
    #       则最终数组形状为 (100*10, action_dim)
    np.savetxt(action_path, np.vstack(actions), delimiter=",", fmt="%.6f")

    # 将所有 step 的状态纵向堆叠，保存为 CSV
    np.savetxt(state_path, np.vstack(states), delimiter=",", fmt="%.6f")

    print(f"saved actions to {action_path}")
    print(f"saved states to {state_path}")


# ===========================================================================
# 主函数 —— 机器人控制主循环
# ===========================================================================
def main() -> None:
    """TRON2 机器人策略客户端的主入口。

    控制流程总览：
    ┌──────────────────────────────────────────────────────────┐
    │  1. 加载配置（YAML + 命令行参数）                          │
    │  2. 初始化机器人环境（Tron2Env）                           │
    │  3. 创建 WebSocket 到策略服务器的连接                       │
    │  4. 启动 Prompt 热加载监听器（支持运行时修改任务指令）        │
    │  5. 主循环（最多 max_steps 步）：                          │
    │     ├─ 从机器人获取观测                                    │
    │     ├─ 发送观测到策略服务器，获取动作预测                    │
    │     ├─ 将动作按分步逐一发送给机器人执行                      │
    │     └─ 按 fps 频率控制循环节奏                             │
    │  6. （可选）保存录制的动作和状态到文件                       │
    └──────────────────────────────────────────────────────────┘
    """

    # ------------------------------------------------------------------
    # 第一步：解析命令行参数并加载部署配置
    # ------------------------------------------------------------------
    args = _parse_args()


    # 提取 client 子配置段落（YAML 中 client: 下面的所有键值对）
    profile_path = select_profile_path(args.profile, args.deploy_config)
    # load_deploy_config 会读取 YAML 文件并返回一个扁平化的配置字典
    config_profile = load_deploy_config(profile_path)
    client_profile = section(config_profile, "client")

    # ------------------------------------------------------------------
    # 安全检查：如果配置中启用了 RTC，拒绝运行
    # RTC（Real-Time Control）模式需要特殊的客户端 pi_client_rtc.py，
    # 因为它采用实时控制协议，和本脚本的同步请求-响应模式不兼容。
    # ------------------------------------------------------------------
    if bool_value(client_profile.get("rtc_enabled", False)):
        raise ValueError(
            "client.rtc_enabled is true. Use examples/tron2/pi_client_rtc.py for RTC deployment."
        )

    # 构建环境配置对象（包含 control_backend、observation_source 等参数）
    env_config = build_env_config(config_profile)

    # ------------------------------------------------------------------
    # 提取关键运行参数
    # ------------------------------------------------------------------
    # max_steps: 最大控制步数，None 表示无限循环（直到手动中断）
    # 优先级：client.max_steps > client.max_inferences > 默认 100
    max_steps = positive_int_or_none(
        client_profile.get("max_steps", client_profile.get("max_inferences", 100)),
        field_name="client.max_steps",
    )

    # fps: 控制频率（帧/秒），即每秒执行多少次"观测→执行"循环
    # 优先使用 client.fps，如未配置则使用环境默认 fps
    fps = float(client_profile.get("fps", env_config.fps))

    # step_period: 每个控制步的时长（秒），用于控制循环节奏
    # 例如 fps=10 时，step_period=0.1 秒
    step_period = 1.0 / fps if fps > 0 else 0.0

    # save_record: 是否将执行过程中的动作和状态保存到文件
    save_record = bool_value(client_profile.get("save_record", False))

    # 打印关键配置信息，方便调试和确认
    print(f"observation_source: {env_config.observation_source}")
    print(f"control_backend: {env_config.control_backend}")
    print(f"publish_rate: {env_config.publish_rate} Hz, fps: {fps} Hz")

    # ------------------------------------------------------------------
    # 第二步：初始化 Prompt 控制器
    # PromptController 允许在运行时通过标准输入动态修改任务指令。
    # 它启动一个后台线程监听 stdin，用户输入新 prompt 后立即生效。
    #
    # 注意：策略模型收到的 prompt 可以指导机器人的行为，例如：
    #   "pick up the red cube" vs "push the blue cylinder forward"
    # 同一个模型可以根据不同的 prompt 执行不同的任务。
    # ------------------------------------------------------------------
    prompt_ctrl = PromptController(
        args.prompt          # 命令行 --prompt 参数
        or client_profile.get("prompt")  # 或者 YAML 中的 client.prompt
    )
    prompt_ctrl.start_stdin_listener()

    # ------------------------------------------------------------------
    # 第三步：初始化数据录制缓冲区
    # record_state:  收集每个 step 的机器人状态（关节角度、末端位姿等）
    # record_action: 收集每个 step 的动作序列（action chunking 的多步动作）
    # ------------------------------------------------------------------
    record_state: list[np.ndarray] = []
    record_action: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # 第四步：创建机器人环境并进入主循环
    # 使用 with 语句确保环境资源（机器人连接、线程等）被正确清理。
    # ------------------------------------------------------------------
    with Tron2Env(env_config) as env:
        # reset() 将机器人恢复到初始状态（归零关节、清除缓冲等）
        env.reset()

        # ------------------------------------------------------------------
        # 创建 WebSocket 策略客户端
        # WebsocketClientPolicy 维护一个到远程 GPU 服务器的长连接。
        # 每次调用 infer() 时，它会：
        #   1. 将观测数据打包为 JSON/protobuf
        #   2. 通过 WebSocket 发送到策略服务器
        #   3. 等待服务器完成神经网络推理
        #   4. 接收并解析返回的动作序列
        # ------------------------------------------------------------------
        ws_client_policy = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host(client_profile),
            port=policy_port(client_profile),
        )

        # ------------------------------------------------------------------
        # 第五步：主控制循环
        # ------------------------------------------------------------------
        # t: 当前步数计数器（从 0 开始）
        t = 0

        # last_action: 上一步执行的最后一个动作（用于安全检查和打印差异）
        # 取前 14 维：左臂 7 个关节 + 右臂 7 个关节
        # 每个机械臂用 8 维表示（7 关节 + 1 夹爪），此处忽略夹爪维度，
        # 只关心关节角度的变化幅度。
        # 如果 env.last_action 为 None（第一步），则设为 None。
        last_action = env.last_action[:14] if env.last_action is not None else None

        # 循环条件：max_steps 为 None 时无限循环，否则执行 max_steps 步
        while max_steps is None or t < max_steps:
            # 打印步数分隔线，方便在日志中定位每一步
            print("\n\n", "#" * 10, "begin infer", t, "#" * 10)

            # ---------------------------------------------------------------
            # 5a. 获取观测 (Observation)
            # ---------------------------------------------------------------
            # 记录开始时间，用于测量获取观测的延迟
            obs_request = time.perf_counter()

            # 从机器人环境获取当前观测
            # obs 是一个字典，典型结构：
            #   {
            #       "state": np.ndarray,    # 机器人状态（关节角度、末端位姿等）
            #       "images": dict,         # 相机图像（如果有的话）
            #       "prompt": str,          # 可选的任务指令
            #   }
            obs = env.get_obs()

            # 计算等待观测的时间（毫秒）
            # 如果机器人通信有延迟，这个值会比较大
            obs_wait_ms = (time.perf_counter() - obs_request) * 1000.0

            # 如果需要录制数据，深拷贝状态并保存（避免后续被修改）
            if save_record:
                record_state.append(obs["state"].copy())

            # ---------------------------------------------------------------
            # 5b. 策略推理 (Policy Inference)
            # ---------------------------------------------------------------
            # format_obs 将原始观测格式化为策略模型期望的输入格式
            # prompt_ctrl.get() 获取当前的 prompt（可能是运行时修改过的）
            #
            # infer_with_timing 执行完整的推理流程并返回：
            #   ans: 推理结果，包含 actions、server_timing、policy_timing
            #        - ans["actions"]: 动作序列，形状 (N_chunks, action_dim)
            #          N_chunks 是策略一次预测的动作分步数（Action Chunking）
            #   timing: 客户端侧的各阶段耗时统计
            # infer_with_timing 返回一个 (answer_dict, timing_dict) 的元组：
            #   ans:    推理结果，包含 actions（动作序列）、server_timing、policy_timing
            #   timing: 客户端侧的 WebSocket 通信耗时统计
            ans, timing = infer_with_timing(
                ws_client_policy,
                format_obs(obs, prompt=prompt_ctrl.get()),
            )

            # 获取服务器返回的计时信息
            # server_timing: 策略服务器内部的耗时（排队 + 推理）
            # policy_timing: 策略模型本身的推理耗时
            server_timing = ans.get("server_timing", {})
            policy_timing = ans.get("policy_timing", {})

            # 打印详细的耗时分析——这对性能调优非常关键
            # - obs_wait: 机器人通信延迟
            # - pack/send/recv_wait/unpack: WebSocket 通信各阶段
            # - server: 服务器侧总耗时（含网络往返）
            # - policy: 纯模型推理耗时
            print(
                "timing: "
                f"obs_wait={obs_wait_ms:.2f}ms, "
                f"ws_total={timing['total_ms']:.2f}ms, "
                f"pack={timing['pack_ms']:.2f}ms, "
                f"send={timing['send_ms']:.2f}ms, "
                f"recv_wait={timing['recv_wait_ms']:.2f}ms, "
                f"unpack={timing['unpack_ms']:.2f}ms, "
                f"payload={timing['payload_kb']:.1f}KB, "
                f"response={timing['response_kb']:.1f}KB, "
                f"server={server_timing.get('infer_ms', 0):.2f}ms, "
                f"policy={policy_timing.get('infer_ms', 0):.2f}ms"
            )

            # ---------------------------------------------------------------
            # 5c. 解析动作序列
            # ---------------------------------------------------------------
            # ans["actions"] 是一个列表，长度为 N_chunks（动作分步数）
            # 每个元素的形状为 (action_dim,)
            # 使用 np.stack 将它们沿第 0 维堆叠成 (N_chunks, action_dim) 的数组
            #
            # 动作维度的典型含义（以双臂为例，每个臂 8 维 = 7 关节 + 1 夹爪）：
            #   actions[i][:7]   → 左臂 7 个关节的目标角度
            #   actions[i][7]    → 左夹爪开合度
            #   actions[i][8:15] → 右臂 7 个关节的目标角度
            #   actions[i][15]   → 右夹爪开合度
            actions = np.stack(ans["actions"], axis=0)

            # 打印第一个和最后一个分步动作，用于检查动作序列的合理性
            # left start/end: 左臂动作的起始和目标
            # right start/end: 右臂动作的起始和目标
            print("left start:", actions[0][:8])
            print("right start:", actions[0][8:])
            print("left end:", actions[-1][:8])
            print("right end:", actions[-1][8:])

            # 录制动作数据（保存整个动作序列，包含所有分步）
            if save_record:
                record_action.append(actions)

            # ---------------------------------------------------------------
            # 5d. 逐步执行动作（Action Chunking 展开）
            # ---------------------------------------------------------------
            # 策略模型一次输出 N 个分步动作（Action Chunking），
            # 这里需要将它们逐一发送给机器人执行，每个分步之间按
            # step_period 的频率控制节奏。
            for action in actions:
                # 构建用于比较的机械臂动作向量：左臂 7 关节 + 右臂 7 关节
                # 注意这里排除了夹爪维度（索引 7 和 15），因为夹爪变化
                # 幅度通常很大（从开到关），会影响关节差异的判断。
                # 索引说明：
                #   action[:7]   → 左臂 7 个关节
                #   action[8:15] → 右臂 7 个关节（跳过索引 7 的左夹爪）
                arm_action = np.concatenate((action[:7], action[8:15]))

                # 安全检查：计算当前动作与上一步动作的差异
                # 如果某个关节的角度变化超过 0.5 弧度（约 28.6°），
                # 打印警告——这有助于发现策略模型的异常输出。
                if last_action is not None:
                    error = np.abs(arm_action - last_action)
                    joint_id = int(np.argmax(error))    # 变化最大的关节索引
                    max_diff = float(error[joint_id])     # 最大变化量
                    if max_diff >= 0.5:
                        print(
                            f"joint {joint_id}'s error is {max_diff:.4f}"
                        )

                # 记录执行开始时间（用于频率控制）
                step_t0 = time.perf_counter()

                # 向机器人发送一个分步动作
                # env.step() 会将动作指令发送给机械臂控制器，并阻塞
                # 直到动作开始执行（或完成，取决于控制后端）
                env.step(action)

                # 更新 last_action 用于下一个分步的差异比较
                last_action = arm_action

                # -----------------------------------------------------------
                # 频率控制（Rate Limiting）
                # -----------------------------------------------------------
                # 计算本次执行实际花费的时间，然后 sleep 剩余时间，
                # 确保每个分步之间的间隔为 step_period。
                #
                # 例如 fps=10 → step_period=0.1s，如果 env.step() 花了
                # 0.03s，则 sleep 0.07s，保证稳定的 10Hz 控制频率。
                sleep_remaining = step_period - (time.perf_counter() - step_t0)
                if sleep_remaining > 0:
                    time.sleep(sleep_remaining)

            # 步数递增
            t += 1

        # ------------------------------------------------------------------
        # 第六步：循环结束，保存录制的数据
        # ------------------------------------------------------------------
        if save_record:
            _save_records(config_profile, record_action, record_state)


# ===========================================================================
# 脚本入口
# ===========================================================================
if __name__ == "__main__":
    main()