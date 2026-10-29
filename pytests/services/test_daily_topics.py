"""Pure stdlib/offline tests: no MaiBot runtime, network, or model provider."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
import asyncio
import sqlite3
import unittest

from src.services.daily_topics import DailyTopicProcessor, TopicStore
from src.services.daily_topics.core import source_batches, validate_topics
from src.services.daily_topics.store import LostLease


def source(number, text="a fact", session="group-a", stamp=10):
    return {
        "source_id": f"message:{number}",
        "message_id": f"platform-{number}",
        "session_id": session,
        "occurred_at": stamp,
        "author": "Alice",
        "text": text,
    }


def topic(evidence, name="Known topic", background="Supported context", identity="topic-1"):
    return {"id": identity, "topic": name, "background": background, "evidence_ids": evidence}


async def fold(topics, parts, info):
    evidence = {entry for old in topics for entry in old["evidence_ids"]}
    evidence.update(part["source_id"] for part in parts)
    return [topic(sorted(evidence))] if evidence else []


class DailyTopicTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "nested" / "topics.sqlite3")
        self.store = TopicStore(self.path)

    def processor(self, summarize=fold, **options):
        return DailyTopicProcessor(self.store, summarize, **options)

    async def process(self, processor=None, day="2026-10-02", cutoff=100, session="group-a", now=1000):
        return await (processor or self.processor()).process(session, day, cutoff, now)

    def test_atomic_import_bootstrap_and_anchor(self):
        boundary = {"epoch": 0, "timezone": "Asia/Shanghai", "local_day": "2026-10-01"}
        self.assertIsNone(self.store.initial_boundary())
        self.assertEqual(self.store.initialize_import(0, boundary), boundary)
        self.assertEqual(self.store.initialize_import(99, {"epoch": 99}), boundary)
        self.assertEqual(self.store.import_cursor(), 0)
        record = source(3, "完整正文\n" * 100)
        anchor = {"row_id": 3, "fingerprint": "stable"}
        self.assertEqual(self.store.ingest_page([record], 3, anchor), 1)
        self.assertEqual(self.store.ingest_page([record], 3, anchor), 0)
        self.store.ingest_page([], 2, {"row_id": 2, "fingerprint": "older"})
        self.assertEqual(self.store.import_cursor(), 3)
        self.assertEqual(self.store.import_anchor(), anchor)
        self.assertEqual(self.store.read_sources("group-a", ["message:3"]), [record])
        self.assertEqual(self.store.read_sources("group-b", ["message:3"]), [])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.ingest_page([source(4), source(3, "different")], 4, {"row_id": 4, "fingerprint": "bad"})
        self.assertEqual(len(self.store.coverage("group-a")), 1)
        self.assertEqual(self.store.import_cursor(), 3)
        self.assertEqual(self.store.import_anchor(), anchor)

    def test_reused_cursor_anchor_is_rejected_atomically(self):
        self.store.ingest_page([source(1)], 1, {"row_id": 1, "fingerprint": "original"})
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.store.ingest_page([source(2)], 1, {"row_id": 1, "fingerprint": "reused"})
        self.assertEqual(self.store.import_anchor()["fingerprint"], "original")
        self.assertEqual(len(self.store.coverage("group-a")), 1)

    async def test_heartbeat_extends_a_slow_model_lease(self):
        self.store.ingest_page([source(1)], 1)
        renewal_seen, finish_model = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        real_renew = self.store.renew

        def record_renew(*args):
            real_renew(*args)
            loop.call_soon_threadsafe(renewal_seen.set)

        self.store.renew = record_renew

        async def slow(topics, parts, info):
            await finish_model.wait()
            return await fold(topics, parts, info)

        task = asyncio.create_task(self.process(self.processor(slow, lease_seconds=0.3)))
        await asyncio.wait_for(renewal_seen.wait(), timeout=3)
        # Beyond the initial lease deadline but within its heartbeat extension.
        contender = TopicStore(self.path).acquire("group-a", "2026-10-02", 100, 1000.31, 20, 16)
        self.assertEqual(contender["status"], "busy")
        finish_model.set()
        self.assertEqual((await task)["status"], "complete")

    async def test_strict_midnight_cutoff_and_order(self):
        self.store.ingest_page(
            [
                source(1, stamp=99.999),
                source(2, stamp=100),
                source(3, stamp=0),
                source(4, stamp=101),
                source(5, stamp=99.999),
            ],
            5,
        )
        calls = []

        async def record(topics, parts, info):
            calls.extend(parts)
            return await fold(topics, parts, info)

        await self.process(self.processor(record))
        self.assertEqual([row["source_id"] for row in calls], ["message:3", "message:1", "message:5"])
        self.assertEqual(
            [row["covered_day"] for row in self.store.coverage("group-a")],
            ["2026-10-02", None, "2026-10-02", None, "2026-10-02"],
        )
        await self.process(day="2026-10-03", cutoff=102)
        self.assertTrue(all(row["covered_day"] for row in self.store.coverage("group-a")))

    async def test_lossless_long_source_many_batches_and_restart(self):
        records = [source(1, "前🙂中\n后" * 40)] + [source(i, str(i) * 9, stamp=i) for i in range(2, 24)]
        self.store.ingest_page(records, 23)
        seen = []

        async def record(topics, parts, info):
            seen.extend(parts)
            return await fold(topics, parts, info)

        result = await self.process(self.processor(record, max_batch_chars=17, max_batches_per_pass=2))
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(self.store.current("group-a"))
        self.assertTrue(all(row["covered_day"] is None for row in self.store.coverage("group-a")))
        self.store = TopicStore(self.path)
        # Changed runtime budget cannot alter the already-frozen split plan.
        processor = self.processor(record, max_batch_chars=1000, max_batches_per_pass=3)
        while result["status"] != "complete":
            result = await self.process(processor)
        for original in records:
            parts = [part for part in seen if part["source_id"] == original["source_id"]]
            self.assertEqual("".join(part["text"] for part in parts), original["text"])
            self.assertEqual([part["part_index"] for part in parts], list(range(len(parts))))
            self.assertEqual(parts[-1]["end_offset"], len(original["text"]))
            self.assertTrue(all(part["total_chars"] == len(original["text"]) for part in parts))
        self.assertTrue(all(row["covered_day"] == "2026-10-02" for row in self.store.coverage("group-a")))
        self.assertEqual(self.store.current("group-a")["version"], 1)

    async def test_failure_preserves_previous_accepted_stage(self):
        self.store.ingest_page([source(1, "abcdefghi")], 1)
        calls = []

        async def fail_second(topics, parts, info):
            calls.append(parts[0]["offset"])
            if info["batch_index"] == 1:
                raise RuntimeError("model unavailable")
            return await fold(topics, parts, info)

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await self.process(self.processor(fail_second, max_batch_chars=3))
        self.assertEqual(calls, [0, 3])
        self.assertEqual(self.store.job_status("group-a", "2026-10-02")["next_batch"], 1)
        self.assertIsNone(self.store.current("group-a"))
        self.assertIsNone(self.store.coverage("group-a")[0]["covered_day"])
        resumed = []

        async def retry(topics, parts, info):
            resumed.append(parts[0]["offset"])
            return await fold(topics, parts, info)

        self.store = TopicStore(self.path)
        await self.process(self.processor(retry))
        self.assertEqual(resumed, [3, 6])

    async def test_late_arrival_after_freeze_deferred_not_lost(self):
        self.store.ingest_page([source(1, "abcdef", stamp=20)], 1)
        processor = self.processor(max_batch_chars=3, max_batches_per_pass=1)
        await self.process(processor)
        self.store.ingest_page([source(2, "late old fact", stamp=5)], 2)
        await self.process(processor)
        current = self.store.current("group-a")
        self.assertEqual(current["topics"][0]["evidence_ids"], ["message:1"])
        self.assertEqual((await self.process(processor))["batches_processed"], 0)
        self.assertIsNone(self.store.coverage("group-a")[1]["covered_day"])
        await self.process(day="2026-10-03")
        self.assertEqual(self.store.current("group-a")["topics"][0]["evidence_ids"], ["message:1", "message:2"])

    async def test_no_change_no_new_version_and_empty_board_supersedes(self):
        self.store.ingest_page([source(1)], 1)
        await self.process()
        original = self.store.current("group-a")
        info_seen = []

        async def unchanged(topics, parts, info):
            info_seen.append(info)
            return topics

        self.store.ingest_page([source(2)], 2)
        result = await self.process(self.processor(unchanged), day="2026-10-03")
        self.assertFalse(result["changed"])
        self.assertEqual(self.store.current("group-a"), original)
        self.assertEqual(info_seen[0]["previous_as_of_day"], "2026-10-02")
        self.assertEqual(self.store.coverage("group-a")[1]["covered_day"], "2026-10-03")

        async def clear(topics, parts, info):
            self.assertEqual(parts, [])
            return []

        result = await self.process(self.processor(clear), day="2026-10-04")
        self.assertTrue(result["changed"])
        self.assertEqual(self.store.current("group-a")["topics"], [])
        self.assertEqual(self.store.current("group-a")["version"], 2)
        self.assertEqual(self.store.current("group-a")["as_of_day"], "2026-10-04")
        result = await self.process(day="2026-10-05")
        self.assertFalse(result["changed"])
        self.assertEqual(self.store.current("group-a")["version"], 2)

    async def test_explicit_correction_replaces_prior_topic_background(self):
        self.store.ingest_page([source(1, "Meeting on Tuesday")], 1)
        await self.process()
        self.store.ingest_page([source(2, "Correction: meeting Wednesday")], 2)

        async def correct(topics, parts, info):
            return [topic([parts[0]["source_id"]], background="Meeting moved to Wednesday")]

        await self.process(self.processor(correct), day="2026-10-03")
        current = self.store.current("group-a")
        self.assertEqual(current["version"], 2)
        self.assertEqual(current["topics"][0]["background"], "Meeting moved to Wednesday")
        self.assertEqual(current["topics"][0]["evidence_ids"], ["message:2"])

    async def test_resumed_old_job_uses_todays_only_release_slot(self):
        self.store.ingest_page([source(1, "abcdef")], 1)
        await self.process(self.processor(max_batch_chars=3, max_batches_per_pass=1))
        self.store.ingest_page([source(2, "later")], 2)
        result = await self.process(day="2026-10-03", cutoff=200)
        self.assertEqual(result["cycle_day"], "2026-10-02")
        self.assertEqual(result["published_cycle_day"], "2026-10-03")
        self.assertEqual(self.store.current("group-a")["as_of_day"], "2026-10-03")
        self.assertEqual(self.store.coverage("group-a")[0]["covered_day"], "2026-10-02")
        result = await self.process(day="2026-10-03", cutoff=200)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["batches_processed"], 0)
        self.assertIsNone(self.store.coverage("group-a")[1]["covered_day"])
        await self.process(day="2026-10-04", cutoff=300)
        self.assertEqual(self.store.coverage("group-a")[1]["covered_day"], "2026-10-04")
        self.assertEqual(self.store.current("group-a")["version"], 2)

    async def test_job_crossing_publish_boundary_consumes_new_days_slot(self):
        self.store.ingest_page([source(1)], 1)
        observed = []

        def current_cycle(epoch):
            observed.append(epoch)
            return "2026-10-03"

        result = await self.process(self.processor(publication_cycle=current_cycle))
        self.assertEqual(result["cycle_day"], "2026-10-02")
        self.assertEqual(result["published_cycle_day"], "2026-10-03")
        self.assertGreaterEqual(observed[0], 1000)
        self.assertEqual(self.store.current("group-a")["as_of_day"], "2026-10-03")
        self.store.ingest_page([source(2)], 2)
        self.assertEqual((await self.process(day="2026-10-03"))["batches_processed"], 0)
        self.assertIsNone(self.store.coverage("group-a")[1]["covered_day"])
        await self.process(day="2026-10-04")
        self.assertEqual(self.store.current("group-a")["version"], 2)

    async def test_invalid_provenance_preserves_published_board(self):
        self.store.ingest_page([source(1), source(9, session="group-b")], 9)
        await self.process()
        original = self.store.current("group-a")
        self.store.ingest_page([source(2)], 10)

        async def invent(topics, parts, info):
            return [topic(["message:9"])]

        with self.assertRaisesRegex(ValueError, "evidence"):
            await self.process(self.processor(invent), day="2026-10-03")
        self.assertEqual(self.store.current("group-a"), original)
        self.assertEqual(self.store.job_status("group-a", "2026-10-03")["next_batch"], 0)
        self.assertIsNone(self.store.coverage("group-a")[1]["covered_day"])

    async def test_future_batch_evidence_is_not_yet_visible(self):
        self.store.ingest_page([source(1, "abc"), source(2, "def")], 2)

        async def future(topics, parts, info):
            return [topic(["message:2"])]

        with self.assertRaisesRegex(ValueError, "unseen"):
            await self.process(self.processor(future, max_batch_chars=3))
        self.assertIsNone(self.store.current("group-a"))

    def test_strict_schema_bounds_and_canonical_order(self):
        a, b = topic(["message:1"], identity="a"), topic(["message:2"], identity="b")
        allowed = {"message:1", "message:2"}
        self.assertEqual(validate_topics([b, a], allowed, 1000), [a, b])
        invalid = [
            None,
            {},
            ["not an object"],
            [dict(a, bogus="ignored")],
            [dict(a, background="")],
            [dict(a, evidence_ids=[])],
            [dict(a, evidence_ids=[3])],
            [a, a],
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_topics(value, allowed, 1000)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_topics([a], allowed, 10)

    async def test_same_session_workers_cannot_double_publish_and_groups_are_isolated(self):
        self.store.ingest_page([source(1), source(2, session="group-b")], 2)
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(topics, parts, info):
            entered.set()
            await release.wait()
            return await fold(topics, parts, info)

        first = asyncio.create_task(self.process(self.processor(blocked)))
        await entered.wait()
        second_store = TopicStore(self.path)
        second = DailyTopicProcessor(second_store, fold)
        self.assertEqual((await self.process(second))["status"], "busy")
        await self.process(second, session="group-b")
        self.assertEqual(second_store.current("group-b")["topics"][0]["evidence_ids"], ["message:2"])
        release.set()
        self.assertEqual((await first)["version"], 1)
        self.assertEqual((await self.process(second))["batches_processed"], 0)
        self.assertEqual(second_store.current("group-a")["version"], 1)

    def test_expired_worker_cannot_stage_renew_finish_or_release_new_lease(self):
        self.store.ingest_page([source(1)], 1)
        old = self.store.acquire("group-a", "2026-10-02", 100, 1000, 1, 16)
        new = TopicStore(self.path).acquire("group-a", "2026-10-02", 100, 1002, 20, 16)
        self.assertEqual(new["status"], "working")
        with self.assertRaises(LostLease):
            self.store.stage("group-a", "2026-10-02", old["token"], 0, [], 1002)
        with self.assertRaises(LostLease):
            self.store.renew("group-a", old["token"], 1002, 20)
        with self.assertRaises(LostLease):
            self.store.finish("group-a", "2026-10-02", old["token"], 0, 0, 1002)
        self.store.release("group-a", old["token"])
        self.assertEqual(self.store.acquire("group-a", "2026-10-02", 100, 1003, 20, 16)["status"], "busy")
        self.store.stage("group-a", "2026-10-02", new["token"], 0, [topic(["message:1"])], 1003)
        self.store.finish("group-a", "2026-10-02", new["token"], 1, 1, 1003)
        self.assertEqual(self.store.current("group-a")["version"], 1)

    async def test_cancelled_callback_releases_lease_without_false_coverage(self):
        self.store.ingest_page([source(1)], 1)
        entered = asyncio.Event()

        async def wait_forever(topics, parts, info):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(self.process(self.processor(wait_forever)))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.store.coverage("group-a")[0]["covered_day"])
        self.assertEqual((await self.process())["status"], "complete")

    async def test_restart_between_last_stage_and_publish_does_not_replay_clear(self):
        self.store.ingest_page([source(1)], 1)
        await self.process()
        acquired = self.store.acquire("group-a", "2026-10-03", 100, 1000, 1, 16)
        self.store.stage("group-a", "2026-10-03", acquired["token"], 0, [], 1000)
        self.store = TopicStore(self.path)

        async def must_not_call(topics, parts, info):
            self.fail("accepted final clear stage must not be rerun")

        result = await self.process(self.processor(must_not_call), day="2026-10-03", now=1002)
        self.assertEqual(result["batches_processed"], 0)
        self.assertEqual(self.store.current("group-a")["topics"], [])
        self.assertEqual(self.store.current("group-a")["version"], 2)

    async def test_old_claim_event_times_are_provided_for_late_corrections(self):
        self.store.ingest_page([source(1, "Meeting moved to Wednesday", stamp=90)], 1)
        await self.process()
        self.store.ingest_page([source(2, "Original Tuesday plan", stamp=5)], 2)
        observed = []

        async def preserve_newer(topics, parts, info):
            observed.append(info["previous_evidence"])
            self.assertEqual(parts[0]["occurred_at"], 5)
            return topics

        await self.process(self.processor(preserve_newer), day="2026-10-03")
        self.assertEqual(observed, [[{"source_id": "message:1", "occurred_at": 90, "author": "Alice"}]])
        self.assertEqual(self.store.current("group-a")["version"], 1)
        self.assertEqual(self.store.read_evidence_metadata("group-b", ["message:1"]), [])

    async def test_failed_finish_rolls_back_version_coverage_and_completion(self):
        self.store.ingest_page([source(1)], 1)
        await self.process()
        original = self.store.current("group-a")
        self.store.ingest_page([source(2)], 2)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TRIGGER fail_finish BEFORE UPDATE OF status ON jobs "
                "WHEN NEW.status='complete' BEGIN SELECT RAISE(ABORT, 'finish interrupted'); END"
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "finish interrupted"):
            await self.process(day="2026-10-03")
        self.assertEqual(self.store.current("group-a"), original)
        self.assertIsNone(self.store.coverage("group-a")[1]["covered_day"])
        job = self.store.job_status("group-a", "2026-10-03")
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["next_batch"], 1)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM versions").fetchone()[0], 1)
            db.execute("DROP TRIGGER fail_finish")

        async def must_not_replay(topics, parts, info):
            self.fail("accepted stage should survive failed final transaction")

        result = await self.process(self.processor(must_not_replay), day="2026-10-03")
        self.assertEqual(result["version"], 2)
        self.assertEqual(self.store.coverage("group-a")[1]["covered_day"], "2026-10-03")

    async def test_initial_empty_day_is_complete_without_fabricated_snapshot(self):
        async def must_not_call(topics, parts, info):
            self.fail("no topic or sources means no model work")

        result = await self.process(self.processor(must_not_call))
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["changed"])
        self.assertIsNone(result["version"])
        self.assertIsNone(self.store.current("group-a"))
        self.assertEqual((await self.process(day="2026-10-01"))["status"], "stale")

    def test_empty_sources_and_source_text_budget(self):
        records = [source(i, "") for i in range(205)] + [source(300, "123456789")]
        batches = source_batches(records, 4)
        self.assertTrue(all(sum(len(part["text"]) for part in batch) <= 4 for batch in batches))
        self.assertTrue(all(len(batch) <= 100 for batch in batches))
        self.assertEqual(len([part for batch in batches for part in batch]), 208)


if __name__ == "__main__":
    unittest.main()
