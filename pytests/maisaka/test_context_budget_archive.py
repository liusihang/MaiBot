"""High-watermark archival preserves complete live protocol groups."""
from datetime import datetime

from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.llm_models.payload_content.context_item import ContextItemMeta, ContextToolCall, FunctionCallItem, ReasoningItem
from src.maisaka.context.messages import ContentContextMessage, ModelOutputContextMessage, ToolResultMessage
from src.maisaka.context.post_processor import process_chat_history_after_cycle


def user(text):
    return ContentContextMessage(MessageSequence([TextComponent(text)]), text, datetime.now())


def turn(name, size=10000):
    reasoning = ModelOutputContextMessage(ReasoningItem(meta=ContextItemMeta.create(logical_turn_id=name)))
    call = ModelOutputContextMessage(FunctionCallItem(
        meta=ContextItemMeta.create(logical_turn_id=name),
        tool_call=ContextToolCall.create(call_id=name + '-call', func_name='query_memory', args={'query': name}),
    ))
    output = ToolResultMessage('x' * size, datetime.now(), name + '-call', name, 'query_memory')
    return [reasoning, call, output]


def test_force_archive_handles_large_tools_with_few_chat_messages():
    first, second = user('original'), user('new task')
    old, active = turn('old'), turn('active')
    history = [first, *old, second, *active]
    result = process_chat_history_after_cycle(
        history, max_context_size=40, force_compaction=True,
        active_logical_turn_id='active', consume_references=False,
    )
    assert result.removed_messages == old
    assert result.history == [first, second, *active]
    assert history == [first, *old, second, *active]  # planning is not a commit
    assert all(got is original for got, original in zip(result.history, [first, second, *active], strict=True))


def test_pending_turn_and_newest_turn_are_protected():
    oldest, pending, newest = turn('old'), turn('pending'), turn('new')
    history = [user('a'), *oldest, user('b'), *pending, user('c'), *newest]
    result = process_chat_history_after_cycle(
        history, max_context_size=40, force_compaction=True,
        pending_call_ids={'pending-call'}, consume_references=False,
    )
    assert result.removed_messages == oldest
    assert all(message in result.history for message in [*pending, *newest])


def test_active_round_survives_interleaved_new_user_arrival():
    active = turn('active')
    later = user('unrelated new arrival')
    result = process_chat_history_after_cycle(
        [user('initial question'), *active[:2], later, active[2]],
        max_context_size=40, force_compaction=True,
        active_logical_turn_id='active', consume_references=False,
    )
    assert not result.removed_messages
    assert all(message in result.history for message in active)
    assert later in result.history


def test_without_high_watermark_old_round_is_unchanged():
    history = [user('question'), *turn('old'), user('followup'), *turn('new')]
    result = process_chat_history_after_cycle(history, max_context_size=40, consume_references=False)
    assert result.history == history
    assert not result.removed_messages
