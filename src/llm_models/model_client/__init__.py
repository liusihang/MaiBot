"""Provider metadata only; Python SDK modules are not loaded at startup."""

from src.config.config import config_manager

_CLIENT_TYPES = {"openai", "openai_responses", "gemini"}


def ensure_client_type_loaded(client_type: str) -> None:
    """Retain configuration/UI discovery without importing an inference SDK."""
    if client_type not in _CLIENT_TYPES:
        return
    from .base_client import ClientProviderRegistration, client_registry

    if client_type not in client_registry.client_registry:
        client_registry.register_provider(
            ClientProviderRegistration(
                client_type=client_type,
                factory=client_registry.get_client_class_instance,
                builtin=True,
                description="Node Pi provider",
            )
        )


def ensure_configured_clients_loaded() -> None:
    for provider in config_manager.get_model_config().api_providers:
        ensure_client_type_loaded(provider.client_type)
