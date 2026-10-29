from datetime import datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import unittest
from src.services import memory_records as records
from src.services import memory_flow_service as flow
from src.A_memorix.core.utils import person_profile_service as profile
from src.maisaka.builtin_tool import remember_memory as tool


def msg(i=1, text="我喜欢喝茶", raw=True):
    from src.common.data_models.message_component_data_model import TextComponent

    return NS(
        message_id=str(i),
        session_id="session",
        platform="qq",
        timestamp=datetime(2026, 10, 28, 12) + timedelta(seconds=i),
        message_info=NS(
            user_info=NS(user_id="123", user_nickname="甲", user_cardname="甲"), group_info=NS(group_id="test-group")
        ),
        raw_message=NS(components=[TextComponent(text=text)]) if raw else None,
        processed_plain_text=text,
        reply_to=None,
    )


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    def test_identity_real_message_shape(self):
        self.assertEqual(flow.ChatSummaryWritebackService._extract_session_group_id(msg()), "test-group")
        self.assertEqual(flow.ChatSummaryWritebackService._extract_session_user_id(msg()), "123")

    async def test_filtered_does_not_advance(self):
        svc = flow.ChatSummaryWritebackService()
        svc._states["session"] = flow.ChatSummaryWritebackState(last_trigger_message_count=64)
        result = NS(success=True, detail="chat_filtered", stored_ids=[], skipped_ids=["id"])
        with (
            patch.object(flow, "count_messages", return_value=100),
            patch.object(flow, "find_messages", return_value=[msg(i) for i in range(1, 101)]),
            patch.object(flow.memory_service, "ingest_summary", new=AsyncMock(return_value=result)),
        ):
            await svc._handle_message(msg(100))
        self.assertEqual(svc._states["session"].last_trigger_message_count, 64)

    async def test_backlog_covers_every_message_and_tail(self):
        svc = flow.ChatSummaryWritebackService()
        svc._states["session"] = flow.ChatSummaryWritebackState(last_trigger_message_count=0)
        result = NS(success=True, detail="", stored_ids=["hash"], skipped_ids=[])
        with (
            patch.object(flow, "count_messages", return_value=80),
            patch.object(flow, "find_messages", return_value=[msg(i) for i in range(1, 81)]),
            patch.object(flow.memory_service, "ingest_summary", new=AsyncMock(return_value=result)) as writer,
        ):
            await svc._handle_message(msg(80), flush=True)
        self.assertEqual([c.kwargs["metadata"]["context_length"] for c in writer.call_args_list], [36, 36, 8])
        self.assertEqual([c.kwargs["metadata"]["trigger_message_count"] for c in writer.call_args_list], [36, 72, 80])
        self.assertTrue(all((c.kwargs["group_id"] == "test-group" for c in writer.call_args_list)))

    async def test_failure_keeps_cursor(self):
        svc = flow.ChatSummaryWritebackService()
        svc._states["session"] = flow.ChatSummaryWritebackState(last_trigger_message_count=0)
        with (
            patch.object(flow, "count_messages", return_value=36),
            patch.object(flow, "find_messages", return_value=[msg(i) for i in range(1, 37)]),
            patch.object(
                flow.memory_service,
                "ingest_summary",
                new=AsyncMock(return_value=NS(success=False, detail="failure", stored_ids=[], skipped_ids=[])),
            ),
        ):
            await svc._handle_message(msg(36))
        self.assertEqual(svc._states["session"].last_trigger_message_count, 0)


class EvidenceTests(unittest.TestCase):
    def test_source_own_text_excludes_quote(self):
        m = msg(text="讲下为什么选日韩")
        m.processed_plain_text = "我喜欢日韩 " + m.processed_plain_text
        self.assertEqual(records.user_text(m), "讲下为什么选日韩")

    def test_question_to_bot_is_not_preference(self):
        m = msg(text="讲下为什么选日韩")
        i = {"evidence_message_id": "1", "evidence_quote": "讲下为什么选日韩", "fact": "甲喜欢日韩"}
        self.assertIsNone(records.direct_self_evidence(i, [m], "甲"))

    def test_other_person_not_self(self):
        m = msg(text="我朋友喜欢日韩")
        i = {"evidence_message_id": "1", "evidence_quote": m.processed_plain_text}
        self.assertIsNone(records.direct_self_evidence(i, [m], "甲"))

    def test_quote_cannot_invent_source(self):
        m = msg()
        i = {"evidence_message_id": "1", "evidence_quote": "我喜欢咖啡"}
        self.assertIsNone(records.direct_self_evidence(i, [m], "甲"))

    def test_direct_self_supported(self):
        m = msg()
        i = {"evidence_message_id": "1", "evidence_quote": "我喜欢喝茶"}
        self.assertIs(records.direct_self_evidence(i, [m], "甲"), m)

    def test_identity_equivalent_dedup(self):
        self.assertEqual(records.fact_key("目标人物的 ID 是“键盘侠”。"), records.fact_key("他的ID是键盘侠"))

    def test_explicit_note_idempotent(self):
        self.assertEqual(
            records.memory_external_id("s", "根除/根神", ["1", "2"]),
            records.memory_external_id("s", "根除/根神", ["2", "1"]),
        )
        self.assertNotEqual(
            records.memory_external_id("s", "根除/根神", ["1"]), records.memory_external_id("other", "根除/根神", ["1"])
        )


class ProfileTests(unittest.TestCase):
    def test_aliases_keep_actual_names_after_rename(self):
        svc = profile.PersonProfileService(
            NS(get_person_profile_alias_override=lambda _: {"aliases": ["旧名", "根除"]})
        )
        with (
            patch.object(svc, "_get_derived_person_aliases", return_value=(["新名"], "新名", [])),
            patch.object(svc, "_collect_alias_suggestions_from_memory", return_value=[]),
        ):
            self.assertEqual(svc.get_person_alias_details("pid")["effective_aliases"], ["新名", "旧名", "根除"])

    def test_retracted_paragraph_not_reused(self):
        svc = profile.PersonProfileService(
            NS(get_fact_claim=lambda _: {"status": "retracted"}, get_paragraph_stale_relation_marks_batch=lambda _: {})
        )
        item = {"hash": "h", "type": "paragraph", "content": "错误事实", "metadata": {"fact_claim_ids": ["c"]}}
        self.assertEqual(svc._filter_stale_paragraph_evidence([item]), [])

    def test_duplicate_fact_not_repeated_in_other_section(self):
        svc = profile.PersonProfileService(NS())
        x = svc._merge_fact_claim_buckets(
            {"stable_facts": ["他的ID是键盘侠"], "identity_settings": []},
            [{"value_text": "目标人物的 ID 是键盘侠", "profile_section": "stable_facts"}],
        )
        self.assertEqual(len(x["stable_facts"]), 1)

    def test_temporal_fact_keeps_its_classification(self):
        svc = profile.PersonProfileService(
            NS(
                get_fact_claim=lambda _: {"status": "active", "profile_section": "recent_interactions"},
                get_paragraph_stale_relation_marks_batch=lambda _: {},
            )
        )
        item = {
            "hash": "h",
            "type": "paragraph",
            "content": "目标人物自称顺之利刃",
            "metadata": {"fact_claim_ids": ["c"]},
        }
        evidence = svc._filter_stale_paragraph_evidence([item])
        x = svc._classify_profile_evidence_rule_based(relation_edges=[], memory_traits=[], vector_evidence=evidence)
        self.assertEqual(x["identity_settings"], [])
        self.assertEqual(len(x["recent_interactions"]), 1)

    def test_group_note_fallback_not_stable_fact(self):
        svc = profile.PersonProfileService(NS())
        x = svc._classify_profile_evidence_rule_based(
            relation_edges=[],
            memory_traits=[],
            vector_evidence=[{"content": "群友让甲说话时接一个梗", "metadata": {"source_type": "explicit_memory"}}],
        )
        self.assertEqual(x["stable_facts"], [])
        self.assertTrue(x["recent_interactions"])


class RememberTests(unittest.IsolatedAsyncioTestCase):
    def context(self):
        from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext

        rt = NS(
            session_id="session",
            chat_stream=NS(group_id="test-group", user_id=""),
            find_source_message_by_id=lambda _: None,
        )
        return NS(
            runtime=rt,
            build_success_result=BuiltinToolRuntimeContext.build_success_result,
            build_failure_result=BuiltinToolRuntimeContext.build_failure_result,
        )

    async def test_filtered_is_not_saved(self):
        with (
            patch.object(tool, "find_messages", return_value=[msg()]),
            patch.object(
                tool.memory_service,
                "ingest_text",
                new=AsyncMock(return_value=NS(success=True, detail="chat_filtered", stored_ids=[], skipped_ids=["id"])),
            ),
        ):
            result = await tool.handle_tool(
                self.context(), NS(arguments={"content": "约定", "evidence_message_ids": ["1"]})
            )
        self.assertFalse(result.success)
        self.assertFalse(result.structured_content["saved"])

    async def test_foreign_chat_evidence_rejected(self):
        m = msg()
        m.session_id = "other"
        ctx = self.context()
        ctx.runtime.find_source_message_by_id = lambda _: m
        with patch.object(tool, "find_messages", return_value=[]):
            result = await tool.handle_tool(ctx, NS(arguments={"content": "约定", "evidence_message_ids": ["1"]}))
        self.assertFalse(result.success)

    async def test_no_receipt_is_not_saved(self):
        with (
            patch.object(tool, "find_messages", return_value=[msg()]),
            patch.object(
                tool.memory_service,
                "ingest_text",
                new=AsyncMock(return_value=NS(success=True, detail="", stored_ids=[], skipped_ids=[])),
            ),
        ):
            result = await tool.handle_tool(
                self.context(), NS(arguments={"content": "约定", "evidence_message_ids": ["1"]})
            )
        self.assertFalse(result.success)

    async def test_success_has_actual_evidence(self):
        with (
            patch.object(tool, "find_messages", return_value=[msg()]),
            patch.object(
                tool.memory_service,
                "ingest_text",
                new=AsyncMock(return_value=NS(success=True, detail="", stored_ids=["hash"], skipped_ids=[])),
            ) as writer,
        ):
            result = await tool.handle_tool(
                self.context(), NS(arguments={"content": "甲喜欢喝茶", "evidence_message_ids": ["1"]})
            )
        self.assertTrue(result.success)
        self.assertTrue(result.structured_content["saved"])
        self.assertEqual(writer.call_args.kwargs["metadata"]["evidence"][0]["text"], "我喜欢喝茶")
        self.assertEqual(writer.call_args.kwargs["group_id"], "test-group")

    async def test_fabricated_alias_rejected(self):
        with patch.object(tool, "find_messages", return_value=[msg()]):
            result = await tool.handle_tool(
                self.context(),
                NS(
                    arguments={
                        "content": "甲的别名",
                        "evidence_message_ids": ["1"],
                        "subject_name": "甲",
                        "aliases": ["不存在的称呼"],
                    }
                ),
            )
        self.assertFalse(result.success)
