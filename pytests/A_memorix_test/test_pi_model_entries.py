"""A_Memorix business entries must dispatch through Pi without Python model I/O."""

from types import SimpleNamespace

import aiohttp
import httpx
import openai
import pytest

from A_memorix.core.embedding import api_adapter
from A_memorix.core.utils import model_routing
from src.config.model_configs import APIProvider, ModelInfo, ModelTaskConfig, TaskConfig
from src.llm_models.model_client.base_client import client_registry
from src.llm_models.utils_model import LLMOrchestrator
from src.services import llm_service, service_task_resolver
from src.services.pi_harness import config as pi_config
from src.services.pi_harness import inference


@pytest.fixture
def pi_rpc(monkeypatch):
    provider = APIProvider(name="fixture", base_url="https://model.invalid/v1", api_key="not-a-credential")
    models = [
        ModelInfo(
            name="Terra",
            model_identifier="terra-fixture",
            api_provider=provider.name,
            model_type="llm",
            extra_params={"reasoning_effort": "high"},
            request_admission={"mode": "strict"},
        ),
        ModelInfo(
            name="vector",
            model_identifier="vector-fixture",
            api_provider=provider.name,
            model_type="embedding",
            extra_params={"dimensions": 99, "task_type": "SEMANTIC_SIMILARITY"},
        ),
        ModelInfo(name="backup", model_identifier="backup-fixture", api_provider=provider.name, model_type="embedding"),
    ]
    task = TaskConfig(model_list=["Terra"], max_tokens=25600, selection_strategy="sequential")
    model_config = SimpleNamespace(
        models=models,
        api_providers=[provider],
        model_task_config=ModelTaskConfig(
            memory=task,
            utils=task,
            embedding=TaskConfig(model_list=["vector", "backup"]),
        ),
    )
    manager = SimpleNamespace(get_model_config=lambda: model_config)
    for module in (api_adapter, pi_config, inference, service_task_resolver):
        monkeypatch.setattr(module, "config_manager", manager)
    monkeypatch.setattr(llm_service, "record_llm_cache_usage", lambda **kwargs: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python model network/legacy path must not be reached")

    for name in ("get_client_class_instance", "get_client_class_instance_async"):
        monkeypatch.setattr(client_registry, name, forbidden)
    monkeypatch.setattr(LLMOrchestrator, "generate_response_async", forbidden)
    monkeypatch.setattr(LLMOrchestrator, "get_embedding", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(openai, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(aiohttp, "ClientSession", forbidden)

    calls = []
    failure = []

    async def request(method, **payload):
        calls.append((method, payload))
        if failure:
            raise failure[0]
        name = (
            payload.get("modelOverride") or payload["config"]["model_task_config"][payload["taskName"]]["model_list"][0]
        )
        model = next(model for model in models if model.name == name)
        route = {"modelName": name, "modelIdentifier": model.model_identifier, "apiProvider": model.api_provider}
        if method == "model.embed_text":
            dimension = payload["options"]["extraParams"].get("dimensions", 3)
            return {"embedding": [1.0] * dimension, "route": route}
        assert method == "model.generate"
        return {
            "message": {"content": [{"type": "text", "text": "fixture result"}], "stopReason": "stop"},
            "route": route,
            **({"responseHandle": "node-fixture-handle"} if payload["options"].get("retainReplay", True) else {}),
        }

    monkeypatch.setattr(inference, "get_harness_client", lambda: SimpleNamespace(request=request))
    return SimpleNamespace(calls=calls, failure=failure, config=model_config, task=task)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_type", ["person_profile", "search_tuning", "summary_import", "episode"])
@pytest.mark.parametrize("single_model", [False, True])
async def test_memory_text_entries_use_pi_rpc_with_original_task_budget(pi_rpc, request_type, single_model):
    selected = "Terra" if single_model else ""
    resolved = model_routing.ResolvedLLMModel("memory", pi_rpc.task, selected)

    result = await model_routing.generate_with_resolved_model(resolved, request_type, "business source data")

    assert result.success is True
    assert result.completion.response == "fixture result"
    assert result.completion.provider_response["replayRetained"] is False
    [(method, payload)] = pi_rpc.calls
    assert method == "model.generate"
    assert payload["requestType"] == request_type
    assert payload["taskName"] == "memory"
    assert payload.get("modelOverride") == (selected or None)
    assert payload["config"]["model_task_config"]["memory"]["max_tokens"] == 25600
    assert payload["config"]["models"][0]["extra_params"]["reasoning_effort"] == "high"
    assert payload["config"]["models"][0]["request_admission"]["mode"] == "strict"
    assert "maxTokens" not in payload["options"]
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "business source data"}]
    assert pi_rpc.task.max_tokens == 25600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, dimensions, expected_dimension",
    [
        ("explicit", None, 3),
        ("explicit", 7, 7),
        ("always", None, 5),
        ("never", 7, 3),
    ],
)
async def test_memory_embedding_dimension_contract_reaches_pi(pi_rpc, mode, dimensions, expected_dimension):
    adapter = api_adapter.EmbeddingAPIAdapter(default_dimension=5, dimension_request_mode=mode)

    result = await adapter._get_embedding_direct("source text", dimensions=dimensions)

    assert len(result) == expected_dimension
    [(method, payload)] = pi_rpc.calls
    assert method == "model.embed_text"
    assert payload["modelOverride"] == "vector"
    assert payload["options"]["extraParamsReplace"] is True
    params = payload["options"]["extraParams"]
    assert params["task_type"] == "SEMANTIC_SIMILARITY"
    if mode == "always" or (mode == "explicit" and dimensions is not None):
        assert params["dimensions"] == expected_dimension
    else:
        assert "dimensions" not in params
    assert adapter._last_success_model_name == "vector"
    assert pi_rpc.config.models[1].extra_params["dimensions"] == 99


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["explicit", "always"])
async def test_embedding_failure_does_not_retry_switch_model_or_cache_guessed_dimension(pi_rpc, monkeypatch, mode):
    monkeypatch.setattr(api_adapter.EmbeddingAPIAdapter, "_GLOBAL_DIMENSION_CACHE", {})
    adapter = api_adapter.EmbeddingAPIAdapter(default_dimension=5, dimension_request_mode=mode)
    pi_rpc.failure.append(RuntimeError("unsupported adapter"))

    with pytest.raises(RuntimeError, match="unsupported adapter"):
        await adapter._detect_dimension()

    assert len(pi_rpc.calls) == 1
    assert pi_rpc.calls[0][1]["modelOverride"] == "vector"
    assert adapter._dimension_detected is False
    assert adapter._dimension is None
    assert adapter._GLOBAL_DIMENSION_CACHE == {}
    assert adapter._last_success_model_name == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_identity", [False, True])
async def test_embedding_rejects_changed_dimension_or_model_identity(pi_rpc, monkeypatch, wrong_identity):
    adapter = api_adapter.EmbeddingAPIAdapter(default_dimension=5, dimension_request_mode="always")

    async def request(method, **payload):
        pi_rpc.calls.append((method, payload))
        return {
            "embedding": [1.0] * (5 if wrong_identity else 3),
            "route": {
                "modelName": "backup" if wrong_identity else "vector",
                "modelIdentifier": "fixture",
                "apiProvider": "fixture",
            },
        }

    monkeypatch.setattr(inference, "get_harness_client", lambda: SimpleNamespace(request=request))
    with pytest.raises(RuntimeError, match="模型身份" if wrong_identity else "请求维度"):
        await adapter._get_embedding_direct("test")
    assert len(pi_rpc.calls) == 1
    assert adapter._last_success_model_name == ""
    assert adapter._last_success_dimension is None
