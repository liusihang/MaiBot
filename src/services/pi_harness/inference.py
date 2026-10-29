"""Typed business facade; every model operation executes in the Node Pi harness.

There is no Python SDK transport, provider retry, model selector, or native
assistant transcript here. Auxiliary replay consists solely of scoped Node
handles; the main conversation never uses this short-operation service.
"""

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import asyncio
import base64
import inspect

from src.common.data_models.llm_service_data_models import (
    LLMAudioTranscriptionResult,
    LLMEmbeddingResult,
    LLMResponseResult,
)
from src.common.logger import get_logger
from src.config.config import config_manager
from src.config.model_configs import APIProvider, ModelInfo, TaskConfig
from src.llm_models.exceptions import EmptyResponseException, ReqAbortException
from src.maisaka.pi_runtime.bridge import PiBridgeError
from src.llm_models.image_normalizer import normalize_context_images
from src.llm_models.model_client.base_client import ResponseRequest, UsageRecord
from src.llm_models.payload_content.context_item import ContextItem, ContextItemBuilder
from src.llm_models.payload_content.resp_format import RespFormat
from src.llm_models.payload_content.tool_option import normalize_tool_options

from .client import get_harness_client
from .config import export_harness_config
from .messages import items_to_messages, response_result, validate_response_outcome

logger = get_logger("pi_inference")


@dataclass(frozen=True)
class PiInputCapabilities:
    """Portable Pi input capabilities, not an instantiated provider client."""

    api_provider: Optional[APIProvider] = None

    @staticmethod
    def get_support_image_formats() -> List[str]:
        return ["jpeg", "jpg", "png", "gif", "webp"]


class PiInferenceClient:
    """Legacy method names, one RPC boundary, no Python inference fallback."""

    def __init__(
        self,
        task_name: str,
        request_type: str = "",
        session_id: str = "",
        *,
        task_config_override: Optional[TaskConfig] = None,
        model_override: Optional[str] = None,
    ) -> None:
        self.task_name = task_name.strip()
        self.request_type = request_type
        self.session_id = session_id
        self._task_config_override = task_config_override.model_copy(deep=True) if task_config_override else None
        self.model_override = model_override
        self.model_for_task = self._get_task_config_or_raise()

    def _get_task_config_or_raise(self) -> TaskConfig:
        if self._task_config_override is not None:
            return self._task_config_override
        tasks = config_manager.get_model_config().model_task_config
        task = getattr(tasks, self.task_name, None)
        if not isinstance(task, TaskConfig):
            raise ValueError("Unknown configured model task: " + self.task_name)
        return task

    def _refresh_task_config(self) -> TaskConfig:
        self.model_for_task = self._get_task_config_or_raise()
        return self.model_for_task

    def _config(self) -> Dict[str, Any]:
        config = export_harness_config()
        # Explicit WebUI/custom task snapshots are data, not a second selector.
        config["model_task_config"][self.task_name] = self._refresh_task_config().model_dump(mode="json")
        return config

    async def _record_usage(
        self,
        payload: Dict[str, Any],
        config: Dict[str, Any],
        started: datetime,
        elapsed: float,
        session_id: str,
        succeeded: bool = True,
    ) -> None:
        """Store business accounting from the actual Node route, never a guessed model."""
        route = payload.get("route")
        if not isinstance(route, dict):
            return
        model_data = next((item for item in config["models"] if item["name"] == route["modelName"]), None)
        if model_data is None:
            raise ValueError("Pi returned an unknown configured model identity")
        from src.llm_models.utils import llm_usage_recorder
        from src.services.model_task_usage_service import record_model_task_usage

        model = ModelInfo.model_validate(model_data)
        try:
            await asyncio.to_thread(
                record_model_task_usage,
                self.task_name,
                model.name,
                route["modelIdentifier"],
                route["apiProvider"],
                payload.get("durationMs", elapsed * 1000) / 1000,
                succeeded,
                started,
            )
            usage = payload.get("usage")
            if isinstance(usage, dict):
                record = UsageRecord(
                    model_name=model.name,
                    provider_name=route["apiProvider"],
                    prompt_tokens=(
                        int(usage.get("input", usage.get("prompt_tokens", usage.get("promptTokenCount", 0))))
                        + int(usage.get("cacheRead", 0))
                        + int(usage.get("cacheWrite", 0))
                    ),
                    completion_tokens=int(usage.get("output", usage.get("completion_tokens", 0))),
                    total_tokens=int(
                        usage.get("totalTokens", usage.get("total_tokens", usage.get("totalTokenCount", 0)))
                    ),
                    prompt_cache_hit_tokens=int(usage.get("cacheRead", 0)),
                    prompt_cache_miss_tokens=int(usage.get("input", 0)) + int(usage.get("cacheWrite", 0)),
                    prompt_cache_reported=bool(
                        usage.get("cacheReported", usage.get("cacheRead") or usage.get("cacheWrite"))
                    ),
                )
                await asyncio.to_thread(
                    llm_usage_recorder.record_usage_to_database,
                    model_info=model,
                    model_usage=record,
                    request_started_at=started,
                    user_id="system",
                    request_type=self.request_type,
                    task_name=self.task_name,
                    session_id=session_id,
                    time_cost=elapsed,
                )
        except Exception:
            # A business statistics failure cannot retry an already completed model.
            logger.exception("记录 Pi 模型调用统计失败")

    async def _rpc(self, method: str, *, session_id: str = "", interrupt_flag: Any = None, **params: Any) -> Any:
        if interrupt_flag is not None and interrupt_flag.is_set():
            raise ReqAbortException("Model request cancelled before Pi dispatch")
        config = self._config()
        started, start_time = datetime.now(), perf_counter()
        effective_session_id = session_id or self.session_id
        request = asyncio.create_task(
            get_harness_client().request(
                method,
                config=config,
                taskName=self.task_name,
                requestType=self.request_type,
                sessionId=effective_session_id,
                **params,
            )
        )
        waiter = asyncio.create_task(interrupt_flag.wait()) if interrupt_flag is not None else None
        try:
            if waiter is not None:
                done, _ = await asyncio.wait((request, waiter), return_when=asyncio.FIRST_COMPLETED)
                if waiter in done and not request.done():
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                    raise ReqAbortException("Model request cancelled through Pi harness")
            payload = await request
            try:
                if method == "model.generate":
                    validate_response_outcome(payload)
            except Exception:
                await self._record_usage(
                    payload, config, started, perf_counter() - start_time, effective_session_id, succeeded=False
                )
                raise
            await self._record_usage(payload, config, started, perf_counter() - start_time, effective_session_id)
            return payload
        except PiBridgeError as exc:
            if exc.details:
                await self._record_usage(
                    exc.details, config, started, perf_counter() - start_time, effective_session_id, succeeded=False
                )
            raise
        finally:
            if waiter is not None:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            if not request.done():
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    def _context_capabilities(self, model_name: Optional[str]) -> Tuple[PiInputCapabilities, Optional[ModelInfo]]:
        config = config_manager.get_model_config()
        task = self._refresh_task_config()
        name = model_name or self.model_override
        if name is None and len(task.model_list) == 1:
            name = task.model_list[0]
        # Multi-model routing is exclusively Node-owned. Do not choose a first
        # model in Python just to feed a legacy context factory's optional hint.
        model = next((item for item in config.models if item.name == name), None)
        if name is not None and model is None:
            raise ValueError("Unknown configured model: " + name)
        provider = next((item for item in config.api_providers if model and item.name == model.api_provider), None)
        return PiInputCapabilities(provider), model

    async def release_responses(self, handles: Sequence[str], *, session_id: str = "") -> int:
        """Release caller-owned native replay handles after the last continuation."""
        if not handles:
            return 0
        result = await get_harness_client().request(
            "model.release",
            handles=list(handles),
            sessionId=session_id or self.session_id,
        )
        return int(result["released"])

    async def generate_response_async(
        self,
        prompt: str,
        *,
        retain_replay: bool = False,
        **kwargs: Any,
    ) -> LLMResponseResult:
        item = ContextItemBuilder().add_text_content(prompt).build()
        return await self.generate_response_with_context_async(lambda _: [item], retain_replay=retain_replay, **kwargs)

    async def generate_response_with_context_async(
        self,
        context_factory: Callable[..., Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        tools: Any = None,
        response_format: Optional[RespFormat] = None,
        raise_when_empty: bool = True,
        interrupt_flag: Any = None,
        session_id: str = "",
        force_stream_mode: bool = False,
        *,
        tool_choice: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        retain_replay: bool = True,
    ) -> LLMResponseResult:
        capabilities, model = self._context_capabilities(model_name)
        value = (
            context_factory(capabilities, model)
            if len(inspect.signature(context_factory).parameters) >= 2
            else context_factory(capabilities)
        )
        items: List[ContextItem] = await value if inspect.isawaitable(value) else value
        # Image decode/resize stays a host media operation and never blocks either
        # loop. It does not perform inference or change existing native handles.
        items = await asyncio.to_thread(normalize_context_images, items)
        options: Dict[str, Any] = {"retainReplay": retain_replay}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["maxTokens"] = max_tokens
        if force_stream_mode:
            options["forceStreamMode"] = True
        if tool_choice is not None:
            options["toolChoice"] = tool_choice
        if extra_params:
            options["extraParams"] = extra_params
        if response_format is not None:
            options["responseFormat"] = {"type": response_format.format_type.value}
            if response_format.schema is not None:
                options["responseFormat"]["json_schema"] = response_format.schema
        if tools:
            options["toolOptions"] = [
                {"name": option.name, "description": option.description, "parameters": option.parameters_schema}
                for option in normalize_tool_options(tools)
            ]
        payload = await self._rpc(
            "model.generate",
            messages=items_to_messages(items),
            options=options,
            modelOverride=model_name or self.model_override,
            session_id=session_id,
            interrupt_flag=interrupt_flag,
        )
        try:
            result = response_result(payload, retain_replay=retain_replay)
            if raise_when_empty and not any(
                block.get("type") in {"thinking", "toolCall"}
                or (block.get("type") == "text" and block.get("text", "").strip())
                for block in payload["message"].get("content", [])
            ):
                raise EmptyResponseException(message="Pi returned no assistant output; request was not retried")
            actual_model = next(
                (item for item in config_manager.get_model_config().models if item.name == result.model_name), None
            )
            if actual_model is None:
                raise ValueError("Pi returned an unknown configured model identity")
            effective = payload.get("effectiveOptions", {})
            task = self.model_for_task
            result.request_snapshot = ResponseRequest(
                model_info=actual_model.model_copy(deep=True),
                context_items=list(items),
                tool_options=normalize_tool_options(tools) if tools else None,
                tool_choice=tool_choice,
                max_tokens=effective.get("maxTokens", max_tokens if max_tokens is not None else task.max_tokens),
                temperature=effective.get("temperature", temperature if temperature is not None else task.temperature),
                response_format=response_format,
                extra_params=dict(extra_params or {}),
            )
            return result
        except Exception:
            # The caller never receives this result, so it cannot own/release
            # its native handle. Cleanup must not replace the conversion error.
            handle = payload.get("responseHandle")
            if isinstance(handle, str) and handle:
                try:
                    await self.release_responses([handle], session_id=session_id)
                except (Exception, asyncio.CancelledError):
                    logger.exception("释放未交付的 Pi 模型响应句柄失败")
            raise

    async def generate_response_from_snapshot_async(self, snapshot: ResponseRequest) -> LLMResponseResult:
        if snapshot.stream_response_handler is not None or snapshot.async_response_parser is not None:
            raise ValueError("Python provider stream/parsing callbacks are not supported by the Pi harness")
        current_model = next(
            (model for model in config_manager.get_model_config().models if model.name == snapshot.model_info.name),
            None,
        )
        # Billing-only changes do not alter inference. Every other configured
        # model value (including nested headers/body/admission) remains pinned.
        route_fields = set(ModelInfo.model_fields) - {"price_in", "price_out", "cache_price_in", "price_periods"}
        if current_model is None or current_model.model_dump(include=route_fields) != snapshot.model_info.model_dump(
            include=route_fields
        ):
            raise ValueError("Snapshot model route changed; explicit migration is required")
        return await self.generate_response_with_context_async(
            lambda _: list(snapshot.context_items),
            temperature=snapshot.temperature,
            max_tokens=snapshot.max_tokens,
            model_name=snapshot.model_info.name,
            tools=snapshot.tool_options,
            response_format=snapshot.response_format,
            tool_choice=snapshot.tool_choice,
            extra_params=snapshot.extra_params,
            interrupt_flag=snapshot.interrupt_flag,
        )

    async def generate_response_for_image(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        *,
        retain_replay: bool = False,
        **kwargs: Any,
    ) -> LLMResponseResult:
        item = ContextItemBuilder().add_text_content(prompt).add_image_content(image_format, image_base64).build()
        return await self.generate_response_with_context_async(lambda _: [item], retain_replay=retain_replay, **kwargs)

    async def generate_response_for_voice(
        self,
        voice_base64: str,
        *,
        session_id: str = "",
        model_name: Optional[str] = None,
        mime_type: str = "audio/wav",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMAudioTranscriptionResult:
        result = await self._rpc(
            "model.transcribe",
            input={"data": voice_base64, "mimeType": mime_type},
            options={"extraParams": dict(extra_params or {})},
            modelOverride=model_name or self.model_override,
            session_id=session_id,
        )
        return LLMAudioTranscriptionResult(text=result["text"])

    async def get_embedding(
        self,
        embedding_input: str,
        *,
        session_id: str = "",
        model_name: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_params_replace: bool = False,
    ) -> LLMEmbeddingResult:
        result = await self._rpc(
            "model.embed_text",
            input=embedding_input,
            options={"extraParams": dict(extra_params or {}), "extraParamsReplace": extra_params_replace},
            modelOverride=model_name or self.model_override,
            session_id=session_id,
        )
        return self._embedding_result(result)

    async def get_image_embedding(
        self,
        image_bytes: bytes,
        *,
        mime_type: str,
        preprocess_version: str,
        session_id: str = "",
        model_name: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMEmbeddingResult:
        result = await self._rpc(
            "model.embed_image",
            input={
                "data": base64.b64encode(image_bytes).decode("ascii"),
                "mimeType": mime_type,
                "preprocessVersion": preprocess_version,
            },
            options={"extraParams": dict(extra_params or {})},
            modelOverride=model_name or self.model_override,
            session_id=session_id,
        )
        return self._embedding_result(result)

    @staticmethod
    def _embedding_result(result: Dict[str, Any]) -> LLMEmbeddingResult:
        route = result["route"]
        return LLMEmbeddingResult(
            embedding=result["embedding"],
            model_name=route["modelName"],
            model_identifier=route["modelIdentifier"],
            api_provider=route["apiProvider"],
            request_protocol_hash=result.get("requestProtocolHash", ""),
            usage=dict(result.get("usage") or {}),
        )

    async def generate_images(
        self,
        prompt: str,
        *,
        images: Any = None,
        model_name: Optional[str] = None,
        session_id: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self._rpc(
            "model.generate_image",
            input={"prompt": prompt, "images": images or []},
            options=options or {},
            modelOverride=model_name or self.model_override,
            session_id=session_id,
        )
