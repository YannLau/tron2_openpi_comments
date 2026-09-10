"""WebSocket 策略服务器：把本机的 policy（动作模型）通过 WebSocket 暴露给远程客户端。

一句话理解
----------
scripts/serve_policy.py 会在本地加载 / 构造一个 policy 对象（统一实现 infer 方法）。
本文件负责把这个 policy 包装成网络服务：

    客户端（机器人程序） --发送观测 obs--> 服务器 --policy.infer(obs)--> 回传动作

通信格式是 “MessagePack + NumPy”（openpi_client.msgpack_numpy）：它能把 dict / NumPy
数组高效地序列化成二进制，比 JSON 省带宽，也不像 pickle 那样允许执行任意代码。

一次典型连接的生命周期
----------------------
    客户端                            服务器
      |  WebSocket 握手               |
      |<-- metadata（第一帧，字典）    |  _handler 开头先主动发服务器元信息
      |--- obs 或 RTC 请求（二进制）-->|
      |<-- action 字典（二进制）------ |  推理结果 + server_timing
      |--- 下一次请求 ... ----------->|  循环往复，直到任意一方断开

协议有两种形态（见 _handler 中的分支）：
1. 旧 / 简单协议：客户端直接把观测 dict 当请求发过来；
2. 新 / RTC 协议：客户端发送 {"__rtc_request": True, "obs": ..., 若干 RTC 参数}，
   用于实时动作分块（Real-Time Chunking，让机器人不必等下一次推理结束才动）。
   服务器会把 RTC 参数按白名单过滤后再转发给 policy。

另外，服务器在“升级为 WebSocket 之前”会先处理一次普通 HTTP 请求：访问 /healthz
返回 200 OK，供容器探针 / 负载均衡做健康检查（见 _health_check）。

新手阅读顺序
------------
1. `__init__`：服务器持有哪些配置（此时还没有开始监听）；
2. `serve_forever` / `run`：如何启动监听；
3. `_handler`（核心）：每个客户端连接进来后做什么——发 metadata、收请求、推理、回包；
4. `_health_check`：健康检查这条旁路。
"""

import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    用 WebSocket 协议对外提供 policy 推理服务；对应的客户端实现见
    packages/openpi-client/src/openpi_client/websocket_client_policy.py。

    Currently only implements the `load` and `infer` methods.

    当前实际只实现了 infer（推理）这条链路：收观测 → 调 policy.infer → 回动作。
    （上面英文里的 “load” 是旧版接口的说法，本文件并没有实现 load。）
    BasePolicy 里的 reset 等接口也没有通过本协议暴露。
    每个连接由 `_handler` 独立处理；同一连接内的请求是串行的（回完一条再收下一条），
    不同连接之间则由 asyncio 并发调度。
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """记录要服务的 policy 与监听参数（此时还没有真正开始监听）。

        Args:
            policy: 真正做推理的对象，例如 policies/policy.py 里的 Policy，
                或它的包装类 PolicyRecorder。只要实现了 infer 即可。
            host: 监听地址。"0.0.0.0" 表示监听所有网卡，同一局域网的其他机器都能连上；
                只想本机访问可改成 "127.0.0.1"。
            port: 监听端口；None 表示交给操作系统随机分配（一般只在测试中使用）。
            metadata: 连接建立后第一时间发给客户端的元信息（例如 action_horizon、
                rtc_enabled，见 scripts/serve_policy.py），客户端据此决定用哪种协议。
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}  # 传 None 时退化为空 dict，避免发送 None
        # websockets 库自身也会打日志；显式设为 INFO，保证“连接建立 / 关闭”等信息可见。
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """同步入口：启动事件循环并一直服务，直到进程被终止（例如 Ctrl+C）。

        普通脚本调用它即可，scripts/serve_policy.py 的最后一行就是这么做的。
        asyncio.run 会创建事件循环、运行 self.run()，结束后负责清理。
        """
        asyncio.run(self.run())

    async def run(self):
        """异步实现：创建 WebSocket 服务器，并阻塞在 serve_forever 上。

        注意区分两个 serve_forever：
        - 本方法内部的 `server.serve_forever()` 是 websockets 库的“服务器级”循环；
        - `self._handler` 是“每个客户端连接”的处理协程，由库在连接到来时自动调用。
        """
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            # 关闭 permessage-deflate 压缩：传的是大块二进制数组，压缩收益有限还增加延迟
            compression=None,
            max_size=None,  # 取消单条消息大小上限（库默认 1MB，图像观测很容易超过）
            process_request=_health_check,  # 握手前先做一次 HTTP 处理，用于 /healthz 健康检查
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """处理“一个客户端连接”的完整生命周期：收请求 → 推理 → 回响应。

        参数 `websocket` 是当前连接对象：websocket.recv() 收一帧，websocket.send() 发一帧。
        下面的 while 循环说明“这个连接会一直服务到客户端断开”。
        """
        logger.info(f"Connection from {websocket.remote_address} opened")
        # 每个连接单独创建一个 Packer（msgpack 打包器），
        # 避免多个连接共用同一个带有内部缓冲区的对象。
        packer = msgpack_numpy.Packer()

        # 协议约定：连上后服务器先主动推一帧 metadata，
        # 客户端的 _wait_for_server() 会先把这一帧读走（见 websocket_client_policy.py）。
        await websocket.send(packer.pack(self._metadata))

        # 保存“上一次请求”的完整耗时，附加到下一次响应中返回
        # （原因见循环末尾的说明）。
        prev_total_time = None
        while True:
            try:
                # start_time 打在 recv 之前：这样 prev_total_time 覆盖
                # “等待请求 + 推理 + 发送响应”的整个服务端周期。
                # 用 monotonic 而非 time.time()：它单调递增，不受系统改时间影响，适合测耗时。
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())

                # Support both old protocol (obs dict) and RTC protocol.
                # 兼容两种协议：靠请求里有没有 "__rtc_request" 这个标记来区分。
                if isinstance(request, dict) and "__rtc_request" in request:
                    obs = request["obs"]  # 真正的观测数据放在 obs 字段里
                    rtc_kwargs = {}
                    # 白名单式提取 RTC 参数：只转发认识的名字，忽略客户端多传的字段，
                    # 避免把未知关键字直接传给 policy.infer 而触发 TypeError。
                    for key in (
                        "inference_delay",  # 推理延迟（步数）：推理期间机器人还会执行多少步
                        "prev_chunk_left_over",  # 上一段动作的剩余部分，用于新旧动作平滑对齐
                        "prev_chunk_left_over_len",  # 上面剩余动作的有效长度（guidance 模式用）
                        "prefix_horizon",  # 旧动作前缀长度（guidance 模式用）
                        "max_guidance_weight",  # guidance 引导强度上限
                        "trained_rtc_mode",  # 走模型训练时学到的原生 RTC 路径（不做 VJP guidance）
                    ):
                        if key in request:
                            rtc_kwargs[key] = request[key]
                    return_raw = request.get("return_raw_actions", False)
                else:
                    # 旧协议：整个 request 就是观测 dict，没有 RTC 参数。
                    obs = request
                    rtc_kwargs = {}
                    return_raw = False

                # 真正调用策略推理。return_raw_actions=True 时，除变换后的动作外
                # 还会返回模型原始输出空间的动作（RTC 客户端需要在原始空间里做合并）。
                infer_time = time.monotonic()
                action = self._policy.infer(obs, return_raw_actions=return_raw, **rtc_kwargs)
                infer_time = time.monotonic() - infer_time

                # 在返回值里附加服务器侧计时，方便客户端监控延迟：
                #   infer_ms      —— 本次 policy.infer 的耗时；
                #   prev_total_ms —— 上一次完整“收请求 → 发响应”周期的耗时
                #                    （见下方说明）。
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    # 本次响应发出前拿不到“本次总耗时”（还要算上 send 的时间），
                    # 所以总耗时只能在“下一次响应”里回传上一次的结果。
                    # 注意：它包含服务器等待客户端下一次请求的空闲时间，
                    # 并非纯网络往返延迟。
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                # send 之后再取时间差，统计里才会包含网络发送耗时。
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                # 客户端正常断开 / 网络中断：属于预期情况，退出循环、结束本协程即可。
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # 推理过程中出现任何异常：先把 Python 堆栈作为“文本帧”发给客户端，
                # 再以 1011 Internal Error 关闭连接，最后 re-raise 让服务端日志也留下记录。
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """WebSocket 握手前的 HTTP 处理钩子（作为 process_request 回调传入）。

    - 路径是 /healthz → 直接返回 200 OK 的普通 HTTP 响应，不建立 WebSocket，
      供容器探针 / 负载均衡判断“服务是否还活着”；
    - 其他路径 → 返回 None，表示不拦截，由 websockets 库继续完成 WebSocket 握手。

    该函数可以是同步的：websockets 允许 process_request 直接返回 Response / None，
    也允许返回 awaitable。
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    # 返回 None＝放行，继续走标准的 WebSocket 握手流程。
    return None
