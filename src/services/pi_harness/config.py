"""In-memory configuration export; never persist this payload in transcripts/logs."""

from typing import Any, Dict

from src.config.config import config_manager


def export_harness_config() -> Dict[str, Any]:
    """Return configured routes verbatim, including auth solely for private IPC."""
    config = config_manager.get_model_config()
    return {
        "models": [model.model_dump(mode="json") for model in config.models],
        "api_providers": [provider.model_dump(mode="json") for provider in config.api_providers],
        "model_task_config": config.model_task_config.model_dump(mode="json"),
    }
