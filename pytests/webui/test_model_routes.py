"""模型路由测试

验证 Gemini 提供商连接测试会使用查询参数传递 API Key，
并且不会回退到 OpenAI 兼容接口使用的 Bearer 认证方式。
"""

import importlib
import base64
import struct
import sys
import zlib
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def load_model_routes(monkeypatch: pytest.MonkeyPatch):
    """在导入路由前 stub 配置与认证依赖模块，避免测试时触发真实初始化。"""

    class DummyConfigManager:
        def get_model_config(self):
            return SimpleNamespace(api_providers=[])

        def register_reload_callback(self, callback: Any) -> None:
            return None

    config_module = ModuleType("src.config.config")
    config_module.__dict__["CONFIG_DIR"] = "."
    config_module.__dict__["config_manager"] = DummyConfigManager()
    monkeypatch.setitem(sys.modules, "src.config.config", config_module)

    dependencies_module = ModuleType("src.webui.dependencies")

    async def require_auth():
        return "test-token"

    dependencies_module.__dict__["require_auth"] = require_auth
    monkeypatch.setitem(sys.modules, "src.webui.dependencies", dependencies_module)

    sys.modules.pop("src.webui.routers.model", None)
    module = importlib.import_module("src.webui.routers.model")
    # These tests exercise auth/routing, not DNS policy. Keep all transport fixtures offline.
    monkeypatch.setattr(module, "validate_public_url", lambda url, **kwargs: url)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["v2p1-raw", "mai_replier_v6", "mai_replier_v7", "mai_replier", "mai_replier_today"])
@pytest.mark.parametrize("response, expected_success", [("我也想出去走走了", True), ("  ", False)])
async def test_replyer_model_uses_chat_reply_instead_of_tools(
    monkeypatch: pytest.MonkeyPatch, identifier: str, response: str, expected_success: bool
) -> None:
    """专用回复模型应使用真实回复模板，正文为空才判失败，不要求工具调用。"""
    model_routes = load_model_routes(monkeypatch)
    monkeypatch.setattr(model_routes, "_get_model_config", lambda name: {
        "model_type": "llm", "model_identifier": identifier, "visual": True,
    })
    captured = {}

    class FakeOrchestrator:
        def __init__(self, model_name: str, *, replyer_test: bool = False) -> None:
            captured["replyer_test"] = replyer_test

        async def generate_response_with_context_async(self, **kwargs: Any):
            captured.update(kwargs)
            return model_routes.LLMResponseResult.from_portable_output(response=response, model_name="V麦回复")

    monkeypatch.setattr(model_routes, "_SingleModelTestOrchestrator", FakeOrchestrator)
    result = await model_routes.test_model_capability(model_routes.ModelTestRequest(model_name="V麦回复"))

    assert result.success is expected_success
    assert result.test_kind == "replyer"
    assert result.visual_tested is False
    assert result.tool_call_ok is False
    assert captured["replyer_test"] is True
    assert captured["tools"] is None
    messages = captured["context_factory"](None)
    from src.llm_models.payload_content.context_item import get_item_text

    assert len(messages) == 2
    assert "麦麦" in get_item_text(messages[0])
    assert "散步" in get_item_text(messages[1])
    assert "{" not in get_item_text(messages[1])
    if not expected_success:
        assert result.error == "模拟 replyer 任务未返回回复正文"


@pytest.mark.asyncio
async def test_replyer_provider_connection_runs_reply_task(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """提供商连接按钮也通过实际回复验证，优先使用 mai_replier，不走通用网络探测。"""
    model_routes = load_model_routes(monkeypatch)
    (tmp_path / "model_config.toml").write_text(
        '''
[[api_providers]]
name = "renamed-provider"
base_url = "https://openzako.ai/v1/"
[[models]]
name = "mai_replier_today"
api_provider = "renamed-provider"
model_identifier = "mai_replier_today"
model_type = "llm"
[[models]]
name = "mai_replier"
api_provider = "renamed-provider"
model_identifier = "mai_replier"
model_type = "llm"
''', encoding="utf-8",
    )
    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))

    async def fake_model_test(request):
        assert request.model_name == "mai_replier"
        return model_routes.ModelTestResponse(
            success=True, model_name=request.model_name, visual_tested=False,
            tool_call_ok=False, test_kind="replyer", latency_ms=12.0,
        )

    async def unexpected_generic_test(**kwargs):
        pytest.fail("麦麦特调回复不应执行通用连接测试")

    monkeypatch.setattr(model_routes, "test_model_capability", fake_model_test)
    monkeypatch.setattr(model_routes, "_test_provider_connection", unexpected_generic_test)
    result = await model_routes.test_provider_connection_by_name(provider_name="renamed-provider")
    assert result["network_ok"] is True
    assert result["api_key_valid"] is True
    assert result["latency_ms"] == 12.0


class FakeResponse:
    """简化版 HTTP 响应对象。"""

    def __init__(self, status_code: int):
        self.status_code = status_code


def build_async_client_factory(
    responses: list[FakeResponse],
    calls: list[dict[str, Any]],
):
    """构造一个可记录请求参数的 AsyncClient 替身。"""

    response_iter = iter(responses)

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def get(
            self,
            url: str,
            headers: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
        ) -> FakeResponse:
            calls.append(
                {
                    "url": url,
                    "headers": headers or {},
                    "params": params or {},
                }
            )
            return next(response_iter)

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_test_provider_connection_uses_query_api_key_for_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini 连接测试应通过查询参数传递 API Key。"""
    model_routes = load_model_routes(monkeypatch)
    calls: list[dict[str, Any]] = []
    fake_client_class = build_async_client_factory(
        responses=[FakeResponse(200), FakeResponse(200)],
        calls=calls,
    )
    monkeypatch.setattr(model_routes.httpx, "AsyncClient", fake_client_class)

    result = await model_routes.test_provider_connection(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="valid-gemini-key",
        client_type="gemini",
    )

    assert result["network_ok"] is True
    assert result["api_key_valid"] is True
    assert len(calls) == 2

    network_call = calls[0]
    validation_call = calls[1]

    assert network_call["url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert network_call["headers"] == {}
    assert network_call["params"] == {}

    assert validation_call["url"] == "https://generativelanguage.googleapis.com/v1beta/models"
    assert validation_call["params"] == {"key": "valid-gemini-key"}
    assert validation_call["headers"] == {"Content-Type": "application/json"}
    assert "Authorization" not in validation_call["headers"]


@pytest.mark.asyncio
async def test_test_provider_connection_uses_bearer_auth_for_openai_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 Gemini 提供商连接测试应继续使用 Bearer 认证。"""
    model_routes = load_model_routes(monkeypatch)
    calls: list[dict[str, Any]] = []
    fake_client_class = build_async_client_factory(
        responses=[FakeResponse(200), FakeResponse(200)],
        calls=calls,
    )
    monkeypatch.setattr(model_routes.httpx, "AsyncClient", fake_client_class)

    result = await model_routes.test_provider_connection(
        base_url="https://example.com/v1",
        api_key="valid-openai-key",
        client_type="openai",
    )

    assert result["network_ok"] is True
    assert result["api_key_valid"] is True
    assert len(calls) == 2

    validation_call = calls[1]

    assert validation_call["url"] == "https://example.com/v1/models"
    assert validation_call["params"] == {}
    assert validation_call["headers"]["Content-Type"] == "application/json"
    assert validation_call["headers"]["Authorization"] == "Bearer valid-openai-key"


@pytest.mark.asyncio
async def test_test_provider_connection_by_name_forwards_provider_client_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """按提供商名称测试连接时，应透传配置中的 client_type。"""
    model_routes = load_model_routes(monkeypatch)
    config_path = tmp_path / "model_config.toml"
    config_path.write_text(
        """
[[api_providers]]
name = "Gemini"
base_url = "https://generativelanguage.googleapis.com/v1beta"
api_key = "valid-gemini-key"
client_type = "gemini"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))

    captured_kwargs: dict[str, Any] = {}

    async def fake_test_provider_connection(**kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return {
            "network_ok": True,
            "api_key_valid": True,
            "latency_ms": 12.34,
            "error": None,
            "http_status": 200,
        }

    monkeypatch.setattr(model_routes, "_test_provider_connection", fake_test_provider_connection)

    result = await model_routes.test_provider_connection_by_name(provider_name="Gemini")

    assert result["network_ok"] is True
    assert result["api_key_valid"] is True
    assert captured_kwargs == {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "valid-gemini-key",
        "client_type": "gemini",
        "allow_configured_private_network": True,
    }


@pytest.mark.asyncio
async def test_test_model_capability_requires_tool_call_and_keeps_visual_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """模型能力测试应要求测试工具调用，并按模型 visual 配置附带视觉测试标记。"""
    model_routes = load_model_routes(monkeypatch)
    config_path = tmp_path / "model_config.toml"
    config_path.write_text(
        """
[[models]]
name = "vision-model"
model_type = "llm"
model_identifier = "vision-model-id"
api_provider = "Fake"
visual = true
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))

    captured: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, model_name: str):
            captured["model_name"] = model_name

        async def generate_response_with_context_async(self, **kwargs: Any):
            captured.update(kwargs)
            return model_routes.LLMResponseResult.from_portable_output(
                response="",
                model_name="vision-model",
                tool_calls=[
                    model_routes.ToolCall(
                        call_id="call-1",
                        func_name=model_routes.MODEL_TEST_TOOL_NAME,
                        args={"status": "ok", "echo": "maibot model test", "saw_image": True},
                    )
                ],
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

    monkeypatch.setattr(model_routes, "_SingleModelTestOrchestrator", FakeOrchestrator)

    result = await model_routes.test_model_capability(model_routes.ModelTestRequest(model_name="vision-model"))

    assert result.success is True
    assert result.visual_tested is True
    assert result.tool_call_ok is True
    assert result.total_tokens == 15
    assert captured["model_name"] == "vision-model"
    assert captured["tools"][0]["name"] == model_routes.MODEL_TEST_TOOL_NAME


@pytest.mark.asyncio
async def test_test_model_capability_tests_embedding_task_model_via_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """嵌入任务中的模型应通过嵌入接口测试，而不是对话接口。"""
    model_routes = load_model_routes(monkeypatch)
    config_path = tmp_path / "model_config.toml"
    config_path.write_text(
        """
[[models]]
name = "embed-model"
model_type = "embedding"
model_identifier = "embed-model-id"
api_provider = "Fake"

[model_task_config.embedding]
model_list = ["embed-model"]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))

    captured: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, model_name: str):
            captured["model_name"] = model_name

        async def get_embedding(self, embedding_input: str, **kwargs: Any):
            captured["embedding_input"] = embedding_input
            return SimpleNamespace(embedding=[0.1] * 8, model_name="embed-model")

        async def generate_response_with_context_async(self, **kwargs: Any):
            raise AssertionError("嵌入模型测试不应调用对话接口")

    monkeypatch.setattr(model_routes, "_SingleModelTestOrchestrator", FakeOrchestrator)

    result = await model_routes.test_model_capability(model_routes.ModelTestRequest(model_name="embed-model"))

    assert result.success is True
    assert result.tool_call_ok is False
    assert result.visual_tested is False
    assert result.response == "嵌入向量维度: 8"
    assert captured["model_name"] == "embed-model"
    assert captured["embedding_input"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_identifier", "image_task_config"),
    [
        ("Qwen/Qwen3-VL-Embedding-8B", ""),
        ("custom-vector-model", '[model_task_config.image_embedding]\nmodel_list = ["my-vector"]'),
    ],
)
async def test_test_model_capability_tests_visual_embedding_via_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    model_identifier: str,
    image_task_config: str,
) -> None:
    """显式图片嵌入类型应发送图片，不按标识或任务名猜测协议。"""
    model_routes = load_model_routes(monkeypatch)
    (tmp_path / "model_config.toml").write_text(
        f'[[models]]\nname = "my-vector"\nmodel_identifier = "{model_identifier}"\n'
        f'api_provider = "Fake"\nmodel_type = "image_embedding"\n\n{image_task_config}',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))
    images: list[bytes] = []

    class FakeOrchestrator:
        def __init__(self, model_name: str):
            assert model_name == "my-vector"

        async def get_image_embedding(self, image_bytes: bytes, **kwargs: Any):
            images.append(image_bytes)
            return SimpleNamespace(embedding=[0.1, 0.2])

        async def get_embedding(self, embedding_input: str, **kwargs: Any):
            raise AssertionError("视觉嵌入模型不应调用文本嵌入接口")

        async def generate_response_with_context_async(self, **kwargs: Any):
            raise AssertionError("视觉嵌入模型不应调用对话接口")

    monkeypatch.setattr(model_routes, "_SingleModelTestOrchestrator", FakeOrchestrator)

    result = await model_routes.test_model_capability(model_routes.ModelTestRequest(model_name="my-vector"))

    assert result.success is True
    assert result.test_kind == "image_embedding"
    assert result.embedding_dimension == 2
    assert len(images) == 3
    assert images[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert images[0] == images[1]
    assert images[0] != images[2]


@pytest.mark.asyncio
async def test_test_model_capability_keeps_chat_test_for_non_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """未配置在嵌入任务中的模型应保持原有对话测试流程。"""
    model_routes = load_model_routes(monkeypatch)
    config_path = tmp_path / "model_config.toml"
    config_path.write_text(
        """
[[models]]
name = "chat-model"
model_type = "llm"
model_identifier = "chat-model-id"
api_provider = "Fake"

[model_task_config.embedding]
model_list = ["other-embed-model"]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_routes, "CONFIG_DIR", str(tmp_path))

    class FakeOrchestrator:
        def __init__(self, model_name: str):
            self.model_name = model_name

        async def get_embedding(self, embedding_input: str, **kwargs: Any):
            raise AssertionError("对话模型测试不应调用嵌入接口")

        async def generate_response_with_context_async(self, **kwargs: Any):
            return model_routes.LLMResponseResult.from_portable_output(
                response="",
                model_name="chat-model",
                tool_calls=[
                    model_routes.ToolCall(
                        call_id="call-1",
                        func_name=model_routes.MODEL_TEST_TOOL_NAME,
                        args={"status": "ok", "echo": "maibot model test", "saw_image": False},
                    )
                ],
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

    monkeypatch.setattr(model_routes, "_SingleModelTestOrchestrator", FakeOrchestrator)

    result = await model_routes.test_model_capability(model_routes.ModelTestRequest(model_name="chat-model"))

    assert result.success is True
    assert result.tool_call_ok is True


def test_model_test_image_base64_is_valid_png(monkeypatch: pytest.MonkeyPatch) -> None:
    """内置视觉测试图应是可被服务商正常解析的有效 PNG。"""
    model_routes = load_model_routes(monkeypatch)
    image_bytes = base64.b64decode(model_routes.MODEL_TEST_IMAGE_BASE64)

    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", image_bytes[16:24]) == (32, 32)

    position = 8
    chunk_types: list[bytes] = []
    while position < len(image_bytes):
        chunk_length = struct.unpack(">I", image_bytes[position : position + 4])[0]
        chunk_type = image_bytes[position + 4 : position + 8]
        chunk_data = image_bytes[position + 8 : position + 8 + chunk_length]
        chunk_crc = struct.unpack(">I", image_bytes[position + 8 + chunk_length : position + 12 + chunk_length])[0]

        assert chunk_crc == zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        chunk_types.append(chunk_type)
        position += 12 + chunk_length

    assert chunk_types[0] == b"IHDR"
    assert b"IDAT" in chunk_types
    assert chunk_types[-1] == b"IEND"


@pytest.fixture
def diagnostic_pi_rpc(monkeypatch):
    """Only fake the IPC boundary, leaving the real facade and route entry intact."""
    import aiohttp
    import openai

    from src.config.model_configs import APIProvider, ModelInfo, ModelTaskConfig, TaskConfig
    from src.llm_models.utils_model import LLMOrchestrator
    from src.services.pi_harness import config as pi_config
    from src.services.pi_harness import inference

    module = load_model_routes(monkeypatch)
    provider = APIProvider(name="fixture", base_url="https://model.invalid/v1", api_key="not-a-credential")
    model = ModelInfo(
        name="selected", model_identifier="chat-fixture", api_provider=provider.name, model_type="llm",
        extra_params={"reasoning_effort": "high"},
    )
    task = TaskConfig(model_list=["selected"], max_tokens=25600)
    config = SimpleNamespace(models=[model], api_providers=[provider], model_task_config=ModelTaskConfig(replyer=task))
    manager = SimpleNamespace(get_model_config=lambda: config)
    monkeypatch.setattr(pi_config, "config_manager", manager)
    monkeypatch.setattr(inference, "config_manager", manager)
    monkeypatch.setattr(module, "_get_model_config", lambda name: model.model_dump(mode="json"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic inference may not call Python network/legacy clients")

    for name in ("get_client_class_instance", "get_client_class_instance_async"):
        monkeypatch.setattr(module.client_registry, name, forbidden)
    monkeypatch.setattr(LLMOrchestrator, "generate_response_with_context_async", forbidden)
    monkeypatch.setattr(LLMOrchestrator, "get_embedding", forbidden)
    monkeypatch.setattr(LLMOrchestrator, "get_image_embedding", forbidden)
    monkeypatch.setattr(module.httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(module.httpx, "Client", forbidden)
    monkeypatch.setattr(openai, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(aiohttp, "ClientSession", forbidden)
    calls = []
    failure = []

    async def request(method, **payload):
        calls.append((method, payload))
        if failure:
            raise failure[0]
        route = {"modelName": model.name, "modelIdentifier": model.model_identifier, "apiProvider": provider.name}
        if method in {"model.embed_text", "model.embed_image"}:
            return {"route": route, "embedding": [0.1, 0.2, 0.3], "requestProtocolHash": "node-protocol-hash"}
        assert method == "model.generate"
        content = (
            [{"type": "text", "text": "周末出去走走吧"}]
            if payload["taskName"] == "replyer"
            else [{"type": "toolCall", "id": "diagnostic-call", "name": module.MODEL_TEST_TOOL_NAME,
                   "arguments": {"status": "ok", "echo": "maibot model test", "saw_image": model.visual}}]
        )
        return {
            "route": route, "message": {"content": content, "stopReason": "toolUse" if content[0]["type"] == "toolCall" else "stop"},
            "responseHandle": "node-diagnostic-handle",
        }

    monkeypatch.setattr(inference, "get_harness_client", lambda: SimpleNamespace(request=request))
    return SimpleNamespace(module=module, config=config, model=model, calls=calls, failure=failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, visual, expected_method, expected_count", [
    ("chat", False, "model.generate", 1),
    ("chat", True, "model.generate", 1),
    ("replyer", False, "model.generate", 1),
    ("embedding", False, "model.embed_text", 3),
    ("image_embedding", False, "model.embed_image", 3),
])
async def test_actual_diagnostic_entry_uses_pi_facade_only(diagnostic_pi_rpc, kind, visual, expected_method, expected_count):
    fixture = diagnostic_pi_rpc
    fixture.model.model_type = "llm" if kind in {"chat", "replyer"} else kind
    fixture.model.model_identifier = "mai_replier_test" if kind == "replyer" else "fixture-model-id"
    fixture.model.visual = visual

    result = await fixture.module.test_model_capability(fixture.module.ModelTestRequest(model_name="selected"))

    assert result.success is True, result.error
    assert len(fixture.calls) == expected_count
    for method, payload in fixture.calls:
        assert method == expected_method
        assert payload["modelOverride"] == "selected"
        assert payload["requestType"] == "webui_model_test"
        task_name = "replyer" if kind == "replyer" else "webui_model_test"
        assert payload["taskName"] == task_name
        expected_task = {
            "model_list": ["selected"], "max_tokens": 512, "temperature": 0.7,
            "selection_strategy": "sequential", "hard_timeout": 90.0,
        }
        actual_task = payload["config"]["model_task_config"][task_name]
        assert {key: actual_task[key] for key in expected_task} == expected_task
    assert fixture.config.model_task_config.replyer.max_tokens == 25600
    assert fixture.model.extra_params["reasoning_effort"] == "high"
    if kind == "chat":
        payload = fixture.calls[0][1]
        assert payload["options"]["maxTokens"] == 512
        assert payload["options"]["toolOptions"][0]["name"] == fixture.module.MODEL_TEST_TOOL_NAME
        content = payload["messages"][0]["content"]
        assert any(part["type"] == "image" for part in content) is visual
    elif kind == "image_embedding":
        images = [base64.b64decode(payload["input"]["data"]) for _, payload in fixture.calls]
        assert images[0] == images[1] and images[0] != images[2]
        assert fixture.calls[0][1]["input"]["preprocessVersion"] == "webui_model_test_v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "embedding", "image_embedding"])
async def test_diagnostic_rpc_failure_is_exposed_without_python_fallback(diagnostic_pi_rpc, kind):
    fixture = diagnostic_pi_rpc
    fixture.model.model_type = kind
    fixture.failure.append(RuntimeError("unsupported Pi adapter"))

    result = await fixture.module.test_model_capability(fixture.module.ModelTestRequest(model_name="selected"))

    assert result.success is False
    assert "unsupported Pi adapter" in result.error
    assert len(fixture.calls) == 1
