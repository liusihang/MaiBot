from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.llm_models.payload_content.context_item import (
    ContextItemMeta,
    ContextToolCall,
    FunctionCallItem,
    FunctionCallOutputItem,
    ReasoningItem,
    ContextTextPart,
    UserMessageItem,
)
from src.maisaka.pi_runtime.codec import encode_items
from src.maisaka.pi_runtime.ledger import EventInput, ThreadLedger
from src.maisaka.pi_runtime.summarization import ThreadSummarizer, public_records


def test_summary_projection_keeps_public_evidence_not_reasoning(tmp_path):
    meta = ContextItemMeta.create(logical_turn_id="turn")
    items = [
        UserMessageItem(meta, (ContextTextPart("actual source text"),)),
        FunctionCallItem(meta, ContextToolCall.create(call_id="c", func_name="read", args={"source": "M1"})),
        FunctionCallOutputItem(meta, "c", "original evidence", "read"),
        ReasoningItem(meta, text_parts=("hidden chain must not be summarized",)),
    ]
    ledger = ThreadLedger(tmp_path / "db")
    snapshot = ledger.append_events(
        "group", [EventInput("e", {"role": "assistant", "native": {"items": encode_items(items)}})]
    )
    projected = public_records(snapshot.events)
    assert "hidden chain" not in str(projected)
    assert projected[0]["event_id"] == "e"
    assert "original evidence" in str(projected) and "M1" in str(projected)


@pytest.mark.asyncio
async def test_summary_requires_complete_bounded_text():
    summarizer = ThreadSummarizer(SimpleNamespace())
    summarizer.provider = AsyncMock(
        return_value={"message": {"stopReason": "stop", "content": [{"type": "text", "text": "summary"}]}}
    )
    assert await summarizer(()) == "summary"
    assert summarizer.provider.call_args.args[0]["tools"] == []
    for reason, text in [("length", "partial"), ("stop", ""), ("stop", "x" * 12001)]:
        summarizer.provider.return_value = {
            "message": {"stopReason": reason, "content": [{"type": "text", "text": text}]}
        }
        with pytest.raises(ValueError):
            await summarizer(())
