from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deploy.qqbot.plugins.qwen3vl_image_provider.plugin import QwenImageProvider


def request():
    return {
        "image_base64": "aW1hZ2U=",
        "mime_type": "image/png",
        "api_provider": {"base_url": "https://embedding.example.com/v1/", "api_key": "must-not-forward", "timeout": 30},
        "model_info": {"name": "qwen-image-local", "model_identifier": "Qwen3-VL-Embedding-2B"},
        "preprocess_version": "image-v2",
    }


def plugin_with_result(result):
    plugin = QwenImageProvider()
    call = AsyncMock(return_value=result)
    plugin._set_context(SimpleNamespace(call_capability=call))
    return plugin, call


@pytest.mark.asyncio
async def test_image_request_uses_only_host_pi_capability_and_preserves_usage():
    plugin, call = plugin_with_result(
        {
            "success": True,
            "embedding": [0.1] * 2048,
            "usage": {"prompt_tokens": 17},
            "request_protocol_hash": "native-protocol-hash",
        }
    )
    await plugin.on_load()
    result = await plugin.embed("image_embedding", request())
    await plugin.on_unload()
    call.assert_awaited_once_with(
        "llm.embed_image",
        timeout_ms=240000,
        task_name="image_embedding",
        model_name="qwen-image-local",
        image_base64="aW1hZ2U=",
        mime_type="image/png",
        preprocess_version="image-v2",
    )
    assert "api_key" not in repr(call.call_args)
    assert not hasattr(plugin, "client")
    assert len(result["embedding"]) == 2048
    assert result["usage"]["total_tokens"] == 17
    assert result["request_protocol_hash"] == "native-protocol-hash"


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.0] * 10, [float("nan")] * 2048, [True] * 2048, "not a vector"])
async def test_invalid_vector_does_not_enter_memory(vector):
    plugin, call = plugin_with_result({"success": True, "embedding": vector})
    with pytest.raises(ValueError, match="2048"):
        await plugin.embed("image_embedding", request())
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_unsupported_input_or_missing_model_never_calls_host():
    plugin, call = plugin_with_result({})
    with pytest.raises(ValueError, match="仅处理图片"):
        await plugin.embed("embedding", request())
    value = request()
    value["mime_type"] = "text/html"
    with pytest.raises(ValueError, match="格式不支持"):
        await plugin.embed("image_embedding", value)
    value = request()
    del value["model_info"]["name"]
    with pytest.raises(ValueError, match="不能根据模型标识猜测"):
        await plugin.embed("image_embedding", value)
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_host_failure_is_exposed_without_direct_http_retry():
    plugin, call = plugin_with_result({"success": False, "error": "adapter unavailable"})
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        await plugin.embed("image_embedding", request())
    assert call.await_count == 1
