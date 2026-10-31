"""Validate event boundaries produced by the same summary call, without generating facts."""

from typing import Any, List


def validated_event_spans(raw: Any, content: str) -> List[str]:
    """Accept only a complete, ordered partition of the exact source text.

    Whitespace between spans is insignificant. Missing, overlapping, reordered or
    rewritten evidence falls back to ordinary event segmentation.
    """
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        return []
    if any(not isinstance(item, str) or not item.strip() for item in raw):
        return []
    spans = [item.strip() for item in raw]
    remaining = content.strip()
    for span in spans:
        if not remaining.startswith(span):
            return []
        remaining = remaining[len(span) :].lstrip()
    return spans if not remaining else []
