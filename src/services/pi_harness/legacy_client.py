"""Compatibility adapter for callers holding old typed client requests.

Registry factories are metadata only. This adapter never creates a provider SDK
or invokes a plugin provider; unsupported capabilities fail in Node explicitly.
"""

from typing import List

from src.config.config import config_manager
from src.config.model_configs import ModelInfo
from src.llm_models.model_client.base_client import (
    APIResponse,
    AudioTranscriptionRequest,
    BaseClient,
    EmbeddingRequest,
    ImageEmbeddingRequest,
    ResponseRequest,
    UsageRecord,
)
from src.llm_models.payload_content.context_item import build_portable_output_items

from .inference import PiInferenceClient


class PiModelClient(BaseClient):
    """Pinned-model facade for legacy diagnostics and plugin-facing request types."""

    def _inference(self, model: ModelInfo, task_name: str) -> PiInferenceClient:
        if model.api_provider != self.api_provider.name:
            raise ValueError("Model request does not belong to this provider")
        config = config_manager.get_model_config()
        current = next((item for item in config.models if item.name == model.name), None)
        if current is None:
            raise ValueError("Legacy snapshot model is no longer configured")
        if (current.api_provider, current.model_identifier, current.extra_params) != (
            model.api_provider,
            model.model_identifier,
            model.extra_params,
        ):
            raise ValueError("Legacy snapshot route changed; explicit migration is required")
        task = getattr(config.model_task_config, task_name)
        return PiInferenceClient(
            task_name,
            request_type="legacy.pi_facade",
            model_override=model.name,
            task_config_override=task.model_copy(update={"model_list": [model.name]}, deep=True),
        )

    async def get_response(self, request: ResponseRequest) -> APIResponse:
        result = await self._inference(request.model_info, "utils").generate_response_from_snapshot_async(request)
        return APIResponse(
            output_items=result.output_items,
            stop_reason=result.stop_reason,
            usage=UsageRecord(
                model_name=result.model_name,
                provider_name=self.api_provider.name,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                prompt_cache_hit_tokens=result.prompt_cache_hit_tokens,
                prompt_cache_miss_tokens=result.prompt_cache_miss_tokens,
                prompt_cache_reported=result.prompt_cache_reported,
            ),
            wire_protocol=result.wire_protocol,
            provider_response=result.provider_response,
            request_snapshot=result.request_snapshot,
        )

    async def get_embedding(self, request: EmbeddingRequest) -> APIResponse:
        result = await self._inference(request.model_info, "embedding").get_embedding(
            request.embedding_input,
            model_name=request.model_info.name,
            extra_params=request.extra_params,
        )
        return APIResponse(
            embedding=result.embedding, request_protocol_hash=result.request_protocol_hash, wire_protocol="pi_harness"
        )

    async def get_image_embedding(self, request: ImageEmbeddingRequest) -> APIResponse:
        result = await self._inference(request.model_info, "image_embedding").get_image_embedding(
            request.image_bytes,
            mime_type=request.mime_type,
            preprocess_version=request.preprocess_version,
            model_name=request.model_info.name,
            extra_params=request.extra_params,
        )
        return APIResponse(
            embedding=result.embedding, request_protocol_hash=result.request_protocol_hash, wire_protocol="pi_harness"
        )

    async def get_audio_transcriptions(self, request: AudioTranscriptionRequest) -> APIResponse:
        result = await self._inference(request.model_info, "voice").generate_response_for_voice(
            request.audio_base64,
            model_name=request.model_info.name,
            extra_params=request.extra_params,
        )
        return APIResponse(
            output_items=build_portable_output_items(content=result.text or ""),
            stop_reason="stop",
            wire_protocol="pi_harness",
        )

    def get_support_image_formats(self) -> List[str]:
        return ["jpeg", "jpg", "png", "gif", "webp"]
