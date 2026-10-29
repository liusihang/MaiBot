from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deploy.qqbot.plugins.qwen3vl_image_provider.plugin import QwenImageProvider


def request():
    return {
        "image_base64": "aW1hZ2U=",
        "mime_type": "image/png",
        "api_provider": {"base_url": "https://embedding.example.com/v1/", "timeout": 30},
        "model_info": {"model_identifier": "Qwen3-VL-Embedding-2B"},
    }


@pytest.mark.asyncio
async def test_image_request_uses_multimodal_messages_and_preserves_usage():
    plugin = QwenImageProvider()
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "data": [{"embedding": [0.1] * 2048}],
            "usage": {"prompt_tokens": 17},
        },
    )
    plugin.client = SimpleNamespace(post=AsyncMock(return_value=response))

    result = await plugin.embed("image_embedding", request())

    call = plugin.client.post.call_args
    assert call.args[0] == "https://embedding.example.com/v1/embeddings"
    assert call.kwargs["json"]["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
    ]
    assert len(result["embedding"]) == 2048
    assert result["usage"]["total_tokens"] == 17


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.0] * 10, [float("nan")] * 2048])
async def test_invalid_vector_does_not_enter_memory(vector):
    plugin = QwenImageProvider()
    plugin.client = SimpleNamespace(
        post=AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                json=lambda: {"data": [{"embedding": vector}]},
            )
        )
    )
    with pytest.raises(ValueError, match="2048"):
        await plugin.embed("image_embedding", request())


@pytest.mark.asyncio
async def test_unsupported_input_never_calls_embedding_server():
    plugin = QwenImageProvider()
    plugin.client = SimpleNamespace(post=AsyncMock())
    with pytest.raises(ValueError, match="仅处理图片"):
        await plugin.embed("embedding", request())
    value = request()
    value["mime_type"] = "text/html"
    with pytest.raises(ValueError, match="格式不支持"):
        await plugin.embed("image_embedding", value)
    plugin.client.post.assert_not_awaited()
