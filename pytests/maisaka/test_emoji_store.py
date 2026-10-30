"""Emoji resource mappings outlive Python processes without becoming model history."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json
import sqlite3
import subprocess
import sys

import pytest

from src.maisaka import emoji_store
from src.maisaka.emoji_store import FIRST_DURABLE_EMOJI_INDEX, MAX_EMOJI_INDEX, EmojiCandidateStore


def test_store_is_lazy_and_default_path_is_instance_scoped(monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("Constructing a store must not resolve or touch the instance directory")

    monkeypatch.setattr(emoji_store, "get_instance_root", forbidden)
    store = EmojiCandidateStore()
    assert not hasattr(store, "history")
    monkeypatch.setattr(emoji_store, "get_instance_root", lambda: tmp_path)
    assert store.allocate("real-chat", ["hash-a"]) == {FIRST_DURABLE_EMOJI_INDEX: "hash-a"}
    assert (tmp_path / "data/pi_harness/emoji_candidates.sqlite3").is_file()


def test_restart_preserves_mapping_scope_and_never_recycles_committed_indices(tmp_path):
    path = tmp_path / "emoji.sqlite3"
    first = EmojiCandidateStore(path).allocate("chat-a", ["hash-a", "hash-b"])
    first_index = min(first)
    assert list(first) == [FIRST_DURABLE_EMOJI_INDEX, FIRST_DURABLE_EMOJI_INDEX + 1]

    restored = EmojiCandidateStore(path)
    second = restored.allocate("chat-b", ["hash-c"])
    assert min(second) > max(first)
    assert restored.resolve("chat-a", first_index) == "hash-a"
    assert restored.resolve("chat-b", first_index) is None
    assert restored.resolve("chat-a", min(second)) is None
    for legacy in (1, 42, FIRST_DURABLE_EMOJI_INDEX - 1):
        assert restored.resolve("chat-a", legacy) is None

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM emoji_candidates")
    third = EmojiCandidateStore(path).allocate("chat-a", ["hash-d"])
    assert min(third) > max(second)
    assert restored.resolve("chat-a", first_index) is None


def test_cross_process_cold_start_allocates_contiguous_unique_batches(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    program = (
        "import json,sys; from src.maisaka.emoji_store import EmojiCandidateStore; "
        "print(json.dumps(EmojiCandidateStore(sys.argv[1]).allocate(sys.argv[2], ['hash'] * 5)))"
    )

    def allocate(index):
        result = subprocess.run(
            [sys.executable, "-c", program, str(path), f"chat-{index}"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return [int(value) for value in json.loads(result.stdout)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(allocate, range(4)))
    all_indices = [index for batch in batches for index in batch]
    assert len(set(all_indices)) == 20
    assert min(all_indices) == FIRST_DURABLE_EMOJI_INDEX
    assert max(all_indices) == FIRST_DURABLE_EMOJI_INDEX + 19
    store = EmojiCandidateStore(path)
    for session_index, batch in enumerate(batches):
        assert batch == list(range(batch[0], batch[0] + 5))
        assert all(store.resolve(f"chat-{session_index}", index) == "hash" for index in batch)
        assert all(store.resolve(f"other-{session_index}", index) is None for index in batch)


@pytest.mark.parametrize("session_id, hashes", [("", ["hash"]), (" ", ["hash"]), ("chat", ["hash", " "])])
def test_invalid_batch_does_not_create_or_allocate_resources(tmp_path, session_id, hashes):
    path = tmp_path / "invalid.sqlite3"
    with pytest.raises(ValueError):
        EmojiCandidateStore(path).allocate(session_id, hashes)
    assert not path.exists()


def test_index_exhaustion_is_atomic_and_never_exceeds_javascript_integer_precision(tmp_path):
    path = tmp_path / "maximum.sqlite3"
    store = EmojiCandidateStore(path)
    store.allocate("chat", ["first"])
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE sqlite_sequence SET seq=? WHERE name='emoji_candidates'", (MAX_EMOJI_INDEX - 1,))
    with pytest.raises(RuntimeError, match="exhausted"):
        store.allocate("chat", ["too-many", "hashes"])
    assert store.allocate("chat", ["last"]) == {MAX_EMOJI_INDEX: "last"}
    with pytest.raises(RuntimeError, match="exhausted"):
        store.allocate("chat", ["past-limit"])
    assert store.resolve("chat", MAX_EMOJI_INDEX + 1) is None


@pytest.mark.parametrize("index", [True, False, 1.0, "1000000000", -1, 0])
def test_resource_lookup_does_not_coerce_non_integer_or_legacy_handles(tmp_path, index):
    path = tmp_path / "unused.sqlite3"
    assert EmojiCandidateStore(path).resolve("chat", index) is None
    assert not path.exists()
