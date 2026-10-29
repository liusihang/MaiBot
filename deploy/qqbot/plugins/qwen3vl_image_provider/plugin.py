"""Qwen3-VL retrieval-v1 image embeddings through the existing local service."""

import math
import httpx
from maibot_sdk import Field, LLMProvider, MaiBotPlugin, PluginConfigBase


class Settings(PluginConfigBase):
    enabled: bool = Field(default=True, description="启用 Qwen3-VL 图片向量协议")
    config_version: str = Field(default="1.0.0", description="配置版本")


class Config(PluginConfigBase):
    plugin: Settings = Field(default_factory=Settings)


class QwenImageProvider(MaiBotPlugin):
    config_model = Config

    async def on_load(self):
        self.client = httpx.AsyncClient(trust_env=False, timeout=240)

    async def on_unload(self):
        await self.client.aclose()

    async def on_config_update(self, scope, config_data, version):
        pass

    @LLMProvider(
        "qwen3vl_image",
        name="Qwen3-VL 图片嵌入",
        version="1.0.0",
        description="既有 retrieval-v1 Qwen 服务的图片 messages 协议",
    )
    async def embed(self, operation, request):
        if operation != "image_embedding":
            raise ValueError("qwen3vl_image 仅处理图片嵌入")
        image = request.get("image_base64")
        mime = request.get("mime_type")
        if not image or mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"):
            raise ValueError("图片嵌入缺少图片或图片格式不支持")
        provider = request["api_provider"]
        model = request["model_info"]
        payload = {
            "model": model["model_identifier"],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}}],
                }
            ],
            "encoding_format": "float",
        }
        response = await self.client.post(
            provider["base_url"].rstrip("/") + "/embeddings",
            json=payload,
            timeout=float(provider.get("timeout") or 240),
        )
        if response.status_code != 200:
            raise RuntimeError(f"Qwen3-VL 图片嵌入 HTTP {response.status_code}")
        data = response.json()
        vector = data["data"][0]["embedding"]
        if len(vector) != 2048 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
            raise ValueError("Qwen3-VL 图片向量不是2048维有限数值")
        usage = data.get("usage") or {}
        return {
            "embedding": vector,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": 0,
                "total_tokens": int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0),
            },
        }


def create_plugin():
    return QwenImageProvider()
