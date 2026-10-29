"""Private durable codec; unlike plugin snapshots, this preserves native replay bytes.

The allowlist is deliberately closed: persisted data cannot import or instantiate
arbitrary Python classes. No provider credentials are part of ContextItems.
"""
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Sequence

import base64

from src.llm_models.payload_content import context_item as ci

_TYPES = {cls.__name__: cls for cls in (
    ci.ContextItemMeta, ci.ContextTextPart, ci.ContextImagePart, ci.ContextRefusalPart,
    ci.ProviderScope, ci.ProviderReplayFragment, ci.ContextToolCall,
    ci.SystemMessageItem, ci.UserMessageItem, ci.ReasoningItem,
    ci.AssistantMessageItem, ci.FunctionCallItem, ci.FunctionCallOutputItem,
    ci.ProviderActivityItem, ci.ProviderOpaqueItem,
)}
_ENUMS = {cls.__name__: cls for cls in (ci.RoleType, ci.ReasoningRepresentation)}
_ITEMS = (ci.SystemMessageItem, ci.UserMessageItem, ci.ReasoningItem,
          ci.AssistantMessageItem, ci.FunctionCallItem, ci.FunctionCallOutputItem,
          ci.ProviderActivityItem, ci.ProviderOpaqueItem)


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if is_dataclass(value) and type(value).__name__ in _TYPES:
        return {"$type": type(value).__name__, "fields": {
            field.name: _encode(getattr(value, field.name)) for field in fields(value)
        }}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported native codec value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("Invalid native codec scalar")
    if set(value) == {"$bytes"}:
        return base64.b64decode(value["$bytes"], validate=True)
    if set(value) == {"$datetime"}:
        return datetime.fromisoformat(value["$datetime"])
    if set(value) == {"$tuple"}:
        return tuple(_decode(item) for item in value["$tuple"])
    if set(value) == {"$enum", "value"} and value["$enum"] in _ENUMS:
        return _ENUMS[value["$enum"]](value["value"])
    if set(value) == {"$type", "fields"} and value["$type"] in _TYPES:
        cls = _TYPES[value["$type"]]
        if set(value["fields"]) != {field.name for field in fields(cls)}:
            raise ValueError("Native codec field set mismatch")
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    raise ValueError("Unknown native codec tag")


def encode_items(items: Sequence[ci.ContextItem]) -> List[Dict[str, Any]]:
    """Encode full items, preserving ordering, identities and native replay."""
    if any(not isinstance(item, _ITEMS) for item in items):
        raise TypeError("Expected ContextItems")
    return [_encode(item) for item in items]


def decode_items(payload: Sequence[Dict[str, Any]]) -> List[ci.ContextItem]:
    """Restore only allowlisted immutable ContextItems (including hash checks)."""
    items = [_decode(item) for item in payload]
    if any(not isinstance(item, _ITEMS) for item in items):
        raise ValueError("Native payload must contain ContextItems")
    return items
