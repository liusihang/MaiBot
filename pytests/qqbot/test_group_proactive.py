from datetime import datetime
import importlib.util
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

spec = importlib.util.spec_from_file_location(
    "group_proactive_test", Path(__file__).resolve().parents[2] / "deploy/qqbot/plugins/group_proactive/plugin.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def ts(text):
    return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def msg(when, mid="human1", user="123", group="test-group", sid="test-session", text="刚聊的游戏更新"):
    return {
        "message_id": mid,
        "session_id": sid,
        "platform": "qq",
        "timestamp": str(when),
        "received_at": str(when),
        "message_info": {"group_info": {"group_id": group}, "user_info": {"user_id": user, "user_nickname": "群友"}},
        "processed_plain_text": text,
        "raw_message": [{"type": "text", "text": text}],
    }


class ProactiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = ts("2026-10-28T14:00:00")
        self.messages = [msg(self.now - 4000)]
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.plugin = self.make_plugin()

    def make_plugin(self):
        p = mod.create_plugin()
        p._plugin_config_instance = mod.Config(
            plugin={"enabled": True},
            target={"group_id": "test-group", "session_id": "test-session", "bot_user_id": "test-bot"},
            timing={"idle_jitter_minutes": 0},
        )
        p._clock = lambda: self.now
        p._ctx = NS(
            message=NS(get_by_time_in_chat=AsyncMock(side_effect=lambda *a, **kw: self.messages)),
            maisaka=NS(proactive=NS(trigger=AsyncMock(return_value={"queued": True, "task_id": "fake-task"}))),
            paths=NS(data_dir=Path(self.root.name)),
            logger=logging.getLogger("test"),
        )
        return p

    async def test_daytime_uses_existing_agent_not_llm_or_send(self):
        result = await self.plugin.check_once()
        self.assertTrue(result["queued"])
        args = self.plugin.ctx.maisaka.proactive.trigger.call_args.kwargs
        self.assertEqual(args["stream_id"], "test-session")
        self.assertEqual(args["priority"], "low")
        self.assertIn("gpt_web_search", args["intent"])
        self.assertIn("核对事件和报道日期", args["intent"])
        self.assertIn("原文链接", args["intent"])
        self.assertIn("严禁政治", args["intent"])
        self.assertIn("reply(msg_id,text)", args["intent"])
        self.assertEqual(self.plugin.ctx.message.get_by_time_in_chat.call_args.kwargs["filter_mai"], True)

    async def test_midnight_and_morning_make_no_model_or_message_calls(self):
        for hour in [0, 1, 6, 8]:
            self.now = ts(f"2026-10-29T{hour:02}:00:00")
            self.assertEqual((await self.plugin.check_once())["reason"], "quiet_hours")
        self.plugin.ctx.message.get_by_time_in_chat.assert_not_called()
        self.plugin.ctx.maisaka.proactive.trigger.assert_not_called()

    async def test_quiet_boundary_nine_and_late_night(self):
        cfg = self.plugin.config.timing
        self.assertFalse(mod.active_window(ts("2026-10-29T08:59:59"), cfg))
        self.assertTrue(mod.active_window(ts("2026-10-29T09:00:00"), cfg))
        self.assertTrue(mod.active_window(ts("2026-10-28T23:54:59"), cfg))
        self.assertFalse(mod.active_window(ts("2026-10-28T23:56:00"), cfg, reserve_round=True))
        self.assertTrue(mod.active_window(ts("2026-10-28T23:56:00"), cfg))

    async def test_not_idle_and_no_history_do_not_queue(self):
        self.messages = [msg(self.now - 300)]
        self.assertEqual((await self.plugin.check_once())["reason"], "not_idle")
        self.plugin._last_human = None
        self.messages = []
        self.assertEqual((await self.plugin.check_once())["reason"], "no_recent_human")

    async def test_one_opportunity_per_silent_period(self):
        await self.plugin.check_once()
        self.now += 10000
        self.assertEqual((await self.plugin.check_once())["reason"], "awaiting_human")
        self.assertEqual(self.plugin.ctx.maisaka.proactive.trigger.await_count, 1)

    async def test_real_user_rearms_but_keeps_ninety_minute_interval(self):
        await self.plugin.check_once()
        self.messages = [msg(self.now + 1, mid="human2")]
        self.now += 5399
        self.assertEqual((await self.plugin.check_once())["reason"], "cooldown")
        self.now += 1
        self.assertTrue((await self.plugin.check_once())["queued"])

    async def test_other_group_and_bot_cannot_rearm(self):
        await self.plugin.check_once()
        for m in [msg(self.now + 1, "other", group="other-group"), msg(self.now + 2, "self", user="test-bot")]:
            await self.plugin.receive(message=m)
        self.assertEqual(self.plugin._last_human["message_id"], "human1")
        self.assertTrue(self.plugin._same_silence())

    async def test_future_send_guard_blocks_at_night(self):
        await self.plugin.check_once()
        self.now = ts("2026-10-29T00:00:00")
        result = await self.plugin.before_send(message=msg(self.now, "out", "test-bot"))
        self.assertEqual(result["action"], "abort")

    async def test_normal_replies_still_allowed_at_night(self):
        await self.plugin.check_once()
        self.now = ts("2026-10-29T00:00:00")
        await self.plugin.receive(message=msg(self.now - 1, "new-user"))
        self.assertEqual(
            (await self.plugin.before_send(message=msg(self.now, "out", "test-bot")))["action"], "continue"
        )

    async def test_expired_task_is_not_sent_later(self):
        await self.plugin.check_once()
        self.now += 301
        self.assertEqual((await self.plugin.before_send(message=msg(self.now, "out", "test-bot")))["action"], "abort")

    async def test_send_is_standalone_and_preserves_all_kwargs(self):
        await self.plugin.check_once()
        outgoing = msg(self.now, "out", "test-bot")
        result = await self.plugin.before_send(
            message=outgoing, set_reply=True, typing=False, show_log=True, storage_message=True, custom="keep"
        )
        self.assertEqual(result["modified_kwargs"]["message"], outgoing)
        self.assertFalse(result["modified_kwargs"]["set_reply"])
        self.assertEqual(result["modified_kwargs"]["custom"], "keep")

    async def test_send_receipt_not_attempt_becomes_dedup_reference(self):
        await self.plugin.check_once()
        out = msg(self.now, "out", "test-bot", text="有个游戏更新 https://example.org/update")
        await self.plugin.after_send(message=out, sent=False)
        self.assertEqual(self.plugin._state["recent_openings"], [])
        await self.plugin.after_send(message=out, sent=True)
        self.assertIn("https://example.org/update", self.plugin._state["recent_openings"][0]["text"])

    async def test_storage_survives_restart_no_duplicate_attempt(self):
        await self.plugin.on_load()
        await self.plugin.check_once()
        await self.plugin.on_unload()
        new = self.make_plugin()
        await new.on_load()
        try:
            self.assertEqual((await new.check_once())["reason"], "awaiting_human")
            new.ctx.maisaka.proactive.trigger.assert_not_called()
        finally:
            await new.on_unload()

    async def test_prompt_status_preserves_protocol_and_is_idempotent(self):
        await self.plugin.check_once()
        kwargs = {
            "session_id": "test-session",
            "item_schema_version": 1,
            "tool_definitions": [{"x": 1}],
            "selection_reason": "original",
            "items": [
                {
                    "item_type": "SystemMessageItem",
                    "parts": [{"type": "text", "text": "人格\n<runtime_clock>现在</runtime_clock>"}],
                }
            ],
        }
        result = (await self.plugin.prepare_request(**kwargs))["modified_kwargs"]
        result2 = (await self.plugin.prepare_request(**result))["modified_kwargs"]
        text = result2["items"][0]["parts"][0]["text"]
        self.assertEqual(text.count(mod.STATUS_START), 1)
        self.assertIn("<runtime_clock>", text)
        self.assertEqual(result2["item_schema_version"], 1)
        self.assertEqual(result2["tool_definitions"], kwargs["tool_definitions"])
        self.assertEqual(result2["selection_reason"], "original")

    async def test_incoming_during_query_does_not_get_overwritten(self):
        async def query(*args, **kwargs):
            await self.plugin.receive(message=msg(self.now, "just-arrived"))
            return self.messages

        self.plugin.ctx.message.get_by_time_in_chat.side_effect = query
        self.assertEqual((await self.plugin.check_once())["reason"], "not_idle")

    async def test_query_error_cannot_bypass_proactive_night_guard(self):
        await self.plugin.check_once()
        self.plugin.ctx.message.get_by_time_in_chat.side_effect = RuntimeError("offline")
        self.assertEqual((await self.plugin.before_send(message=msg(self.now, "out", "test-bot")))["action"], "abort")

    async def test_uncertain_queue_result_not_retried_for_same_silence(self):
        self.plugin.ctx.maisaka.proactive.trigger.return_value = {"success": False, "error": "timeout"}
        with self.assertRaises(RuntimeError):
            await self.plugin.check_once()
        self.assertEqual((await self.plugin.check_once())["reason"], "awaiting_human")

    async def test_epoch_formats_are_beijing_consistent(self):
        target = ts("2026-10-28T14:00:00")
        self.assertEqual(mod.epoch(str(target)), target)
        self.assertEqual(mod.epoch("2026-10-28T14:00:00"), target)
        self.assertEqual(mod.epoch("2026-10-28T06:00:00Z"), target)

    async def test_sent_opening_prompts_stop_and_real_user_removes_injection(self):
        await self.plugin.check_once()
        await self.plugin.after_send(message=msg(self.now, "out", "test-bot"), sent=True)
        kwargs = {
            "session_id": "test-session",
            "item_schema_version": 1,
            "items": [{"item_type": "SystemMessageItem", "parts": [{"type": "text", "text": "人格"}]}],
        }
        result = await self.plugin.prepare_request(**kwargs)
        self.assertIn("当前有效=false", result["modified_kwargs"]["items"][0]["parts"][0]["text"])
        await self.plugin.receive(message=msg(self.now + 1, "new-human"))
        self.assertNotIn("modified_kwargs", await self.plugin.prepare_request(**kwargs))

    async def test_native_context_protocol_accepts_hook_output(self):
        from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType
        from src.llm_models.payload_content.context_protocol import ContextProtocolMode
        from src.plugin_runtime.hook_payloads import serialize_prompt_items, deserialize_prompt_items

        await self.plugin.check_once()
        original = [
            ContextItemBuilder().set_role(RoleType.System).add_text_content("人格").build(),
            ContextItemBuilder().add_text_content("群友的真消息").build(),
        ]
        output = (
            await self.plugin.prepare_request(
                items=serialize_prompt_items(original),
                item_schema_version=1,
                session_id="test-session",
                tool_definitions=[],
            )
        )["modified_kwargs"]
        restored = deserialize_prompt_items(
            output["items"],
            item_schema_version=output["item_schema_version"],
            mode=ContextProtocolMode.REQUEST_CONTEXT,
            original_items=original,
        )
        self.assertEqual(len(restored), 2)
        self.assertIn(mod.STATUS_START, restored[0].parts[0].text)
        self.assertEqual(restored[1].parts[0].text, "群友的真消息")

    async def test_native_dispatcher_combines_status_and_live_clock_without_losing_schema(self):
        from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType
        from src.llm_models.payload_content.context_protocol import ContextProtocolMode
        from src.plugin_runtime.hook_payloads import serialize_prompt_items, deserialize_prompt_items
        from src.plugin_runtime.host.component_registry import HookHandlerEntry
        from src.plugin_runtime.host.hook_dispatcher import HookDispatcher
        from src.plugin_runtime.host.hook_spec_registry import HookSpec

        clock_spec = importlib.util.spec_from_file_location(
            "live_clock_for_test",
            Path(__file__).resolve().parents[2] / "deploy/qqbot/plugins/replyer_autonomy/plugin.py",
        )
        clock_module = importlib.util.module_from_spec(clock_spec)
        clock_spec.loader.exec_module(clock_module)
        clock = clock_module.create_plugin()
        await self.plugin.check_once()
        entries = [
            HookHandlerEntry(
                name=name,
                component_type="HOOK_HANDLER",
                plugin_id=pid,
                metadata={"hook": "maisaka.planner.before_request", "mode": "blocking"},
            )
            for pid, name in [
                ("local.group-proactive", "prepare_request"),
                ("local.replyer-autonomy", "configure_reply_handoff"),
            ]
        ]

        async def invoke(method, plugin_id, handler_name, arguments, **options):
            p = self.plugin if plugin_id == "local.group-proactive" else clock
            return NS(payload=await getattr(p, handler_name)(**arguments))

        supervisor = NS(
            group_name="third_party", invoke_plugin=invoke, component_registry=NS(get_hook_handlers=lambda _: entries)
        )
        dispatcher = HookDispatcher()
        dispatcher.register_hook_spec(
            HookSpec(name="maisaka.planner.before_request", parameters_schema={}, allow_kwargs_mutation=True)
        )
        original = [ContextItemBuilder().set_role(RoleType.System).add_text_content("人格").build()]
        try:
            result = await dispatcher.invoke_hook(
                "maisaka.planner.before_request",
                supervisors=[supervisor],
                items=serialize_prompt_items(original),
                item_schema_version=1,
                tool_definitions=[],
                session_id="test-session",
                selected_history_count=0,
                built_message_count=1,
                selection_reason="test",
            )
            self.assertFalse(result.errors)
            restored = deserialize_prompt_items(
                result.kwargs["items"],
                item_schema_version=result.kwargs.get("item_schema_version"),
                mode=ContextProtocolMode.REQUEST_CONTEXT,
                original_items=original,
            )
            self.assertIn(mod.STATUS_START, restored[0].parts[0].text)
            self.assertIn("<runtime_clock>", restored[0].parts[0].text)
            self.assertEqual(result.kwargs["session_id"], "test-session")
        finally:
            await dispatcher.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
