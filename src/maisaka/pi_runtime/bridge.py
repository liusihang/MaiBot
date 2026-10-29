"""Asynchronous JSONL transport to the real Pi Agent sidecar."""

from typing import Any, Awaitable, Callable, Dict, Sequence

import asyncio
import json


class PiBridgeError(RuntimeError):
    """Safe typed sidecar failure; never silently fall back to another model."""

    def __init__(self, message: str, *, code: str = "", details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class PiBridge:
    def __init__(self, command: Sequence[str], callback: Callable[[str, Dict[str, Any]], Awaitable[Any]]) -> None:
        self.command = tuple(command)
        self.callback = callback
        self._process = None
        self._reader_task = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._callbacks: Dict[str, asyncio.Task] = {}
        self._counter = 0
        self._write_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Whether this transport can accept a new operation (not a retry)."""
        return (
            self._process is not None
            and self._process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def start(self) -> None:
        if self._process is not None:
            return
        # stderr is intentionally not relayed: provider/tool errors may contain
        # private chat material. RPC errors carry safe structured diagnostics.
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=64 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read())

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self._process is None or self._process.returncode is not None:
            raise PiBridgeError("Pi sidecar is not running")
        async with self._write_lock:
            self._process.stdin.write((json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode())
            await self._process.stdin.drain()

    async def request(self, method: str, **params: Any) -> Any:
        await self.start()
        self._counter += 1
        request_id = f"host-{self._counter}"
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _dispatch(self, payload: Dict[str, Any]) -> None:
        callback_id = payload["id"]
        try:
            result = await self.callback(payload["method"], payload.get("params", {}))
            await self._send({"id": callback_id, "result": result})
        except asyncio.CancelledError:
            # A canceled side effect remains unknown in the host outbox; this
            # response is not evidence that the external effect did not happen.
            await self._send({"id": callback_id, "error": {"code": "cancelled", "message": "Host callback cancelled"}})
            raise
        except Exception as exc:
            await self._send(
                {
                    "id": callback_id,
                    "error": {"code": type(exc).__name__, "message": f"Host callback failed ({type(exc).__name__})"},
                }
            )
        finally:
            self._callbacks.pop(callback_id, None)

    async def _read(self) -> None:
        error = PiBridgeError("Pi sidecar closed its transport")
        try:
            while line := await self._process.stdout.readline():
                payload = json.loads(line)
                if "method" in payload:
                    if payload["method"] == "cancel":
                        task = self._callbacks.get(payload["params"]["callbackId"])
                        if task is not None:
                            task.cancel()
                    elif "id" in payload:
                        task = asyncio.create_task(self._dispatch(payload))
                        self._callbacks[payload["id"]] = task
                    else:
                        raise PiBridgeError("Unknown Pi notification")
                else:
                    future = self._pending.get(payload.get("id"))
                    if future is not None and not future.done():
                        if "error" in payload:
                            error_payload = payload["error"]
                            if isinstance(error_payload, dict):
                                details = error_payload.get("details")
                                future.set_exception(
                                    PiBridgeError(
                                        str(error_payload.get("message", "Pi operation failed")),
                                        code=str(error_payload.get("code", "")),
                                        details=details if isinstance(details, dict) else None,
                                    )
                                )
                            else:
                                future.set_exception(PiBridgeError("Pi operation failed"))
                        else:
                            future.set_result(payload.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = PiBridgeError(f"Pi protocol failed: {type(exc).__name__}")
        finally:
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(error)
            for task in list(self._callbacks.values()):
                task.cancel()

    async def close(self) -> None:
        if self._process is None:
            return
        if self._process.returncode is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                # The reader can observe EOF before asyncio records exit status.
                pass
        await self._process.wait()
        if self._reader_task is not None:
            await asyncio.gather(self._reader_task, return_exceptions=True)
        await asyncio.gather(*list(self._callbacks.values()), return_exceptions=True)
        self._process = None
