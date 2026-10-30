"""One auxiliary Pi process, owned by a broker loop shared by Bot and WebUI.

Only the broker owns subprocess transports and asyncio Futures. Host loops wrap
concurrent futures; cancelling one request cannot cancel a different host loop's
request. Shutdown is terminal and drains cancellation before closing transport.
"""

from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

import asyncio
import atexit
import threading
import uuid

from src.maisaka.pi_runtime.bridge import PiBridge


class PiHarnessClient:
    def __init__(self, command: Optional[Sequence[str]] = None) -> None:
        self.command = tuple(
            command or ("node", str(Path(__file__).resolve().parents[3] / "runtime/pi_agent/sidecar.mjs"))
        )
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bridge: Optional[PiBridge] = None
        self._transport_lock: Optional[asyncio.Lock] = None
        self._active: Set[asyncio.Task] = set()
        self._closed = False
        self._closing: Optional[Future] = None

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Pi inference client is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._serve, name="pi-inference-rpc", daemon=True)
                self._thread.start()
        self._ready.wait()

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._transport_lock = asyncio.Lock()
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _reject_callback(self, method: str, params: Dict[str, Any]) -> Any:
        raise PermissionError("Auxiliary inference cannot execute host tools: " + method)

    async def _request(self, method: str, params: Dict[str, Any]) -> Any:
        current = asyncio.current_task()
        self._active.add(current)
        try:
            async with self._transport_lock:
                if self._bridge is not None and not self._bridge.is_running:
                    # A NEW operation may recover a dead transport. Never replay
                    # any failed request; process-scoped old handles must fail.
                    await self._bridge.close()
                    self._bridge = None
                if self._bridge is None:
                    self._bridge = PiBridge(self.command, self._reject_callback)
                await self._bridge.start()
                bridge = self._bridge
            try:
                return await bridge.request(method, **params)
            except asyncio.CancelledError:
                # Transport cancellation alone cannot abort an in-flight provider.
                # Send its scoped operation ID, never retry generation on failure.
                try:
                    await asyncio.wait_for(
                        bridge.request(
                            "model.cancel",
                            operationId=params["operationId"],
                            sessionId=params.get("sessionId", ""),
                        ),
                        timeout=5,
                    )
                except Exception:
                    # A failed cancellation acknowledgement must not turn into a
                    # generation retry. Shutdown will close the process as well.
                    pass
                raise
            finally:
                if method not in {"model.cancel", "model.release"}:
                    # The RPC has delivered an ACK/error (or cancellation was
                    # requested). Retire operation-id dedup, NOT replay handles.
                    # Native response lifetime belongs to explicit Node handles.
                    try:
                        await asyncio.wait_for(
                            bridge.request(
                                "model.release",
                                operationIds=[params["operationId"]],
                                sessionId=params.get("sessionId", ""),
                            ),
                            timeout=5,
                        )
                    except Exception:
                        # Do not mask a completed inference or retry it just
                        # because optional dedup bookkeeping failed.
                        pass
        finally:
            self._active.discard(current)

    def _submit(self, method: str, params: Dict[str, Any]) -> Future:
        self._start()
        with self._lock:
            if self._closed:
                raise RuntimeError("Pi inference client is closed")
            return asyncio.run_coroutine_threadsafe(self._request(method, params), self._loop)

    async def request(self, method: str, **params: Any) -> Any:
        params.setdefault("operationId", uuid.uuid4().hex)
        # Thread creation and its readiness wait never block either host loop.
        submission = asyncio.create_task(asyncio.to_thread(self._submit, method, params))
        try:
            future = await asyncio.shield(submission)
        except asyncio.CancelledError:
            # Cancellation during broker startup must not orphan a dispatched
            # request in the worker thread after the caller has disappeared.
            future = await submission
            future.cancel()
            raise
        return await asyncio.wrap_future(future)

    async def _shutdown(self) -> None:
        active = list(self._active)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        if self._bridge is not None:
            await self._bridge.close()
            self._bridge = None

    def _begin_close(self) -> Optional[Future]:
        with self._lock:
            self._closed = True
            if self._thread is None:
                return None
        self._ready.wait()
        with self._lock:
            if self._closing is None:
                self._closing = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            return self._closing

    def _stop_thread(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Pi inference broker did not stop")

    async def close(self) -> None:
        closing = await asyncio.to_thread(self._begin_close)
        if closing is not None:
            await asyncio.shield(asyncio.wrap_future(closing))
            await asyncio.to_thread(self._stop_thread)

    def close_at_exit(self) -> None:
        closing = self._begin_close()
        if closing is not None:
            try:
                closing.result(timeout=7)
                self._stop_thread()
            except Exception:
                # No alternate model transport is started during interpreter exit.
                pass


_client = PiHarnessClient()
atexit.register(_client.close_at_exit)


def get_harness_client() -> PiHarnessClient:
    return _client
