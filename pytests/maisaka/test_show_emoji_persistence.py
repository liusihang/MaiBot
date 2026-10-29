"""Real built-in emoji entry points use durable scoped business mappings only."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import asyncio
import threading

from PIL import Image, ImageDraw

import pytest

from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import send_emoji, show_emoji_list
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.emoji_store import FIRST_DURABLE_EMOJI_INDEX, MAX_EMOJI_INDEX, EmojiCandidateStore


def _context(path, session_id="chat-a", store=None):
    return BuiltinToolRuntimeContext(
        SimpleNamespace(passive_replyer_enabled=False),
        SimpleNamespace(session_id=session_id, _focus_group=None, _chat_history=[]),
        emoji_store=store if store is not None else EmojiCandidateStore(path),
    )


@pytest.fixture
def collage_inputs(monkeypatch):
    output = BytesIO()
    Image.new("RGB", (24, 24), color="red").save(output, format="PNG")
    png = output.getvalue()
    monkeypatch.setattr(show_emoji_list, "resolve_enable_visual_planner", lambda: True)
    monkeypatch.setattr(
        show_emoji_list,
        "emoji_manager",
        SimpleNamespace(
            emojis=[SimpleNamespace(file_hash="hash-a"), SimpleNamespace(file_hash="hash-b")],
        ),
    )
    monkeypatch.setattr(show_emoji_list, "sample", lambda values, size: values[:size])
    monkeypatch.setattr(show_emoji_list, "_get_emoji_candidate_count", lambda: 2)

    async def load(emoji):
        return png

    monkeypatch.setattr(show_emoji_list, "_load_emoji_bytes", load)
    return png


@pytest.mark.asyncio
async def test_show_restart_and_clear_python_history_keep_the_same_emoji_mapping(monkeypatch, tmp_path, collage_inputs):
    path = tmp_path / "emoji.sqlite3"
    context = _context(path)
    invocation = ToolInvocation(tool_name="show_emoji_list", arguments={}, call_id="show-1")
    result = await show_emoji_list.handle_tool(context, invocation)
    assert result.success is True
    [message] = result.post_history_messages
    assert message.session_id == "chat-a"
    assert message.emoji_hashes == {FIRST_DURABLE_EMOJI_INDEX: "hash-a", FIRST_DURABLE_EMOJI_INDEX + 1: "hash-b"}
    assert str(FIRST_DURABLE_EMOJI_INDEX) in result.content

    loaded = []

    async def load_hash(emoji_hash):
        loaded.append(emoji_hash)
        return emoji_hash

    monkeypatch.setattr(BuiltinToolRuntimeContext, "load_emoji_attachment", staticmethod(load_hash))
    restarted = _context(path)
    assert restarted.runtime._chat_history == []
    assert await restarted._resolve_emoji_attachment(FIRST_DURABLE_EMOJI_INDEX) == "hash-a"
    newer = await show_emoji_list.handle_tool(restarted, invocation)
    assert min(newer.post_history_messages[0].emoji_hashes) > max(message.emoji_hashes)
    assert await restarted._resolve_emoji_attachment(FIRST_DURABLE_EMOJI_INDEX) == "hash-a"

    # Even a copied or tampered in-memory history cannot authorize another chat.
    foreign = _context(path, "chat-b")
    foreign.runtime._chat_history.append(message)
    with pytest.raises(ValueError, match="当前会话中没有持久映射"):
        await foreign._resolve_emoji_attachment(FIRST_DURABLE_EMOJI_INDEX)
    message.emoji_hashes[1] = "hash-wrong"
    restarted.runtime._chat_history.append(message)
    with pytest.raises(ValueError, match="旧进程编号不可恢复"):
        await restarted._resolve_emoji_attachment(1)
    assert loaded == ["hash-a", "hash-a"]


@pytest.mark.asyncio
async def test_failed_collage_does_not_reuse_its_committed_numbers(monkeypatch, tmp_path, collage_inputs):
    path = tmp_path / "emoji.sqlite3"
    context = _context(path)

    def fail_collage(*args):
        raise RuntimeError("image decode failed")

    monkeypatch.setattr(show_emoji_list, "_merge_emoji_tiles", fail_collage)
    result = await show_emoji_list.handle_tool(
        context,
        ToolInvocation(tool_name="show_emoji_list", arguments={}, call_id="show-failed"),
    )
    assert result.success is False
    assert result.post_history_messages == []
    assert "image decode failed" in result.error_message
    next_mapping = await context.allocate_emoji_candidates(["hash-c"])
    assert min(next_mapping) == FIRST_DURABLE_EMOJI_INDEX + 2


@pytest.mark.asyncio
async def test_storage_failure_never_returns_a_collage_or_falls_back_to_memory(monkeypatch, tmp_path, collage_inputs):
    class FailedStore:
        def allocate(self, *args):
            raise RuntimeError("disk unavailable")

        def resolve(self, *args):
            raise RuntimeError("disk unavailable")

    context = _context(tmp_path / "unused.sqlite3", store=FailedStore())

    async def forbidden_load(*args):
        pytest.fail("No collage may be built before durable allocation succeeds")

    monkeypatch.setattr(show_emoji_list, "_load_emoji_bytes", forbidden_load)
    result = await show_emoji_list.handle_tool(
        context,
        ToolInvocation(tool_name="show_emoji_list", arguments={}, call_id="show-failed"),
    )
    assert result.success is False
    assert "disk unavailable" in result.error_message
    with pytest.raises(RuntimeError, match="disk unavailable"):
        await context._resolve_emoji_attachment(FIRST_DURABLE_EMOJI_INDEX)


@pytest.mark.asyncio
async def test_resource_sqlite_work_is_off_loop_for_allocation_and_lookup(monkeypatch, tmp_path):
    loop_thread = threading.get_ident()
    worker_threads = []

    class TrackingStore(EmojiCandidateStore):
        def allocate(self, *args):
            worker_threads.append(threading.get_ident())
            return super().allocate(*args)

        def resolve(self, *args):
            worker_threads.append(threading.get_ident())
            return super().resolve(*args)

    context = _context(tmp_path / "emoji.sqlite3", store=TrackingStore(tmp_path / "emoji.sqlite3"))

    async def load_hash(emoji_hash):
        return emoji_hash

    monkeypatch.setattr(BuiltinToolRuntimeContext, "load_emoji_attachment", staticmethod(load_hash))
    mapping = await context.allocate_emoji_candidates(["hash-a"])
    assert await context._resolve_emoji_attachment(min(mapping)) == "hash-a"
    assert len(worker_threads) == 2
    assert all(thread != loop_thread for thread in worker_threads)


def test_same_store_is_safe_across_two_independent_event_loops(tmp_path):
    path = tmp_path / "emoji.sqlite3"
    store = EmojiCandidateStore(path)

    def run(session_id):
        context = _context(path, session_id, store)
        return asyncio.run(context.allocate_emoji_candidates([session_id] * 3))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["bot-loop", "webui-loop"]))
    assert len(set(results[0]) | set(results[1])) == 6
    assert all(store.resolve("bot-loop", index) == "bot-loop" for index in results[0])
    assert all(store.resolve("webui-loop", index) is None for index in results[0])


@pytest.mark.parametrize("index", [FIRST_DURABLE_EMOJI_INDEX, MAX_EMOJI_INDEX])
def test_full_durable_index_label_fits_the_existing_collage_tile(monkeypatch, collage_inputs, index):
    actual_text = ImageDraw.ImageDraw.text
    labels = []

    def draw_text(draw, xy, text, *args, **kwargs):
        if text == str(index):
            box = draw.textbbox(xy, text, font=kwargs["font"])
            labels.append((text, box))
        return actual_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", draw_text)
    collage = send_emoji._merge_emoji_tiles([collage_inputs], start_index=index)
    assert len(labels) == 1
    assert labels[0][0] == str(index)
    left, top, right, bottom = labels[0][1]
    with Image.open(BytesIO(collage)) as rendered:
        assert rendered.size == (256, 256)
        assert 0 <= left < right <= rendered.width
        assert 0 <= top < bottom <= rendered.height


@pytest.mark.asyncio
async def test_removed_emoji_resource_is_not_replaced_by_another_candidate(monkeypatch, tmp_path):
    import importlib

    manager_module = importlib.import_module("src.emoji_system.emoji_manager")
    context = _context(tmp_path / "emoji.sqlite3")
    mapping = await context.allocate_emoji_candidates(["deleted-hash"])
    requested = []

    def missing(emoji_hash):
        requested.append(emoji_hash)
        return None

    monkeypatch.setattr(manager_module, "emoji_manager", SimpleNamespace(get_emoji_by_hash=missing))
    with pytest.raises(ValueError, match="已不可用"):
        await context._resolve_emoji_attachment(min(mapping))
    assert requested == ["deleted-hash"]


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [True, False, 1.0, "1000000000", -1, 0])
async def test_attachment_entry_rejects_invalid_index_without_touching_store(tmp_path, index):
    class NoAccessStore:
        def resolve(self, *args):
            pytest.fail("Invalid attachment indices must not reach storage")

    context = _context(tmp_path / "unused.sqlite3", store=NoAccessStore())
    with pytest.raises(ValueError, match="正整数序号"):
        await context._resolve_emoji_attachment(index)
