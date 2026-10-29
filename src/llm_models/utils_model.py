"""Compatibility names for the Node-owned Pi inference harness.

Python no longer selects, retries, serializes provider requests or owns native
assistant history. Configuration helpers remain available for business code.
"""

from typing import Any, Tuple

from src.common.data_models.llm_service_data_models import LLMResponseResult  # noqa: F401
from src.config.config import config_manager
from src.config.model_configs import APIProvider, ModelInfo, TaskConfig
from src.services.pi_harness.inference import PiInferenceClient


class LLMOrchestrator(PiInferenceClient):
    """Legacy public method names; all operations cross the private Pi RPC."""


class TempMethodsLLMUtils:
    @staticmethod
    def get_model_info_by_name(model_name: str) -> ModelInfo:
        """根据模型名称获取模型信息。

        Args:
            model_name: 模型名称

        Returns:
            ModelInfo: 模型信息。

        Raises:
            ValueError: 未找到指定模型。
        """
        for model in config_manager.get_model_config().models:
            if model.name == model_name:
                return model
        raise ValueError(f"未找到名为 '{model_name}' 的模型")

    @staticmethod
    def get_provider_by_name(provider_name: str) -> APIProvider:
        """根据提供商名称获取提供商信息。

        Args:
            provider_name: 提供商名称

        Returns:
            APIProvider: API 提供商信息。

        Raises:
            ValueError: 未找到指定提供商。
        """
        for provider in config_manager.get_model_config().api_providers:
            if provider.name == provider_name:
                return provider
        raise ValueError(f"未找到名为 '{provider_name}' 的API提供商")


class LLMRequest(LLMOrchestrator):
    """兼容旧调用方的 LLM 请求入口。

    新代码应优先使用 ``LLMOrchestrator`` 或服务层 ``LLMServiceClient``；
    该类保留旧版 ``model_set=TaskConfig`` 的构造方式，并在运行时解析
    对应的最新任务配置名称。
    """

    def __init__(self, model_set: TaskConfig, request_type: str = "") -> None:
        """初始化旧版 LLM 请求对象。

        Args:
            model_set: 旧调用方传入的任务配置对象。
            request_type: 当前请求的业务类型标识。
        """
        self._task_config_name = self._resolve_task_config_name(model_set)
        super().__init__(task_name=self._task_config_name, request_type=request_type)

    @staticmethod
    def _build_task_config_signature(task_config: TaskConfig) -> Tuple[Any, ...]:
        """构造任务配置签名。

        Args:
            task_config: 任务配置对象。

        Returns:
            Tuple[Any, ...]: 可用于匹配热重载前后任务配置的签名。
        """
        return (
            tuple(task_config.model_list),
            task_config.max_tokens,
            task_config.temperature,
            task_config.selection_strategy,
            task_config.hard_timeout,
        )

    @classmethod
    def _resolve_task_config_name(cls, model_set: TaskConfig) -> str:
        """根据旧版 TaskConfig 对象解析任务配置名称。

        Args:
            model_set: 旧调用方传入的任务配置对象。

        Returns:
            str: 对应 ``model_task_config`` 下的字段名。

        Raises:
            ValueError: 未能找到匹配任务配置时抛出。
        """
        target_signature = cls._build_task_config_signature(model_set)
        model_task_config = config_manager.get_model_config().model_task_config
        for attr_name in dir(model_task_config):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(model_task_config, attr_name)
            if not isinstance(attr_value, TaskConfig):
                continue
            if attr_value is model_set or cls._build_task_config_signature(attr_value) == target_signature:
                return attr_name
        raise ValueError("无法根据旧版 model_set 解析任务配置名称")
