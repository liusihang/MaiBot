"""Qwen3-VL retrieval-v1 embeddings via the host's Node Pi adapter only."""

from typing import Any, Dict

import math

from maibot_sdk import Field, LLMProvider, MaiBotPlugin, PluginConfigBase


class Settings(PluginConfigBase):
    enabled: bool = Field(default=True, description="启用 Qwen3-VL 图片向量协议")
    config_version: str = Field(default="1.0.0", description="配置版本")


class Config(PluginConfigBase):
    plugin: Settings = Field(default_factory=Settings)


class QwenImageProvider(MaiBotPlugin):
    config_model = Config

    async def on_load(self) -> None:
        """The host owns credentials, model HTTP transport and cancellation."""

    async def on_unload(self) -> None:
        """No plugin-side HTTP client or model process needs closing."""

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        pass

    @LLMProvider(
        "qwen3vl_image",
        name="Qwen3-VL 图片嵌入",
        version="1.1.0",
        description="通过宿主 Pi 侧适配器执行 retrieval-v1 Qwen 图片协议",
    )
    async def embed(self, operation: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve the plugin boundary without recursion into plugin providers.

        llm.embed_image resolves the configured host model and calls Node's
        explicit qwen3vl-retrieval-v1 adapter, not LLMOrchestrator/PluginLLMClient.
        Neither provider URLs nor credentials from this legacy request are sent
        to the capability: the host's current model configuration is authoritative.
        """
        if operation != "image_embedding":
            raise ValueError("qwen3vl_image 仅处理图片嵌入")
        image = request.get("image_base64")
        mime = request.get("mime_type")
        if not image or mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"):
            raise ValueError("图片嵌入缺少图片或图片格式不支持")
        model_name = request.get("model_info", {}).get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("图片嵌入缺少宿主已配置的模型名称，不能根据模型标识猜测路由")
        result = await self.ctx.call_capability(
            "llm.embed_image",
            timeout_ms=240000,
            task_name="image_embedding",
            model_name=model_name,
            image_base64=image,
            mime_type=mime,
            preprocess_version=request.get("preprocess_version") or "qwen3vl-retrieval-v1",
        )
        if not isinstance(result, dict) or result.get("success") is not True:
            detail = result.get("error", "无有效宿主结果") if isinstance(result, dict) else "无有效宿主结果"
            raise RuntimeError(f"Pi Qwen3-VL 图片嵌入失败: {detail}")
        vector = result.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != 2048
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("Qwen3-VL 图片向量不是2048维有限数值")
        usage = result.get("usage") or {}
        return {
            "embedding": vector,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0),
            },
            "request_protocol_hash": result.get("request_protocol_hash", ""),
        }


def create_plugin() -> QwenImageProvider:
    return QwenImageProvider()
