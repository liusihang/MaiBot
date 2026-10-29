import asyncio
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import unittest

from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType
from src.llm_models.payload_content.context_protocol import ContextProtocolMode
from src.plugin_runtime.hook_payloads import serialize_prompt_items, deserialize_prompt_items


spec = importlib.util.spec_from_file_location(
    "clock_candidate", Path(__file__).resolve().parents[2] / "deploy/qqbot/plugins/replyer_autonomy/plugin.py"
)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


def fixture():
    return [
        ContextItemBuilder()
        .set_role(RoleType.System)
        .add_text_content("核心规则。文中提及 <runtime_clock> 只是说明，不能删掉后面的正文。\n人格和工具约定。")
        .build(),
        ContextItemBuilder().add_text_content("时间：2026-10-27 00:00:00").build(),
        ContextItemBuilder().add_text_content("[msg_id:test]真实聊天").build(),
        ContextItemBuilder().add_text_content("时间：2026-10-28 14:00:00").build(),
        ContextItemBuilder().add_text_content("选择工具或发言").build(),
    ]


class ClockTests(unittest.TestCase):
    def test_timezone_date_and_weekday(self):
        items = candidate.inject_runtime_clock(
            serialize_prompt_items(fixture()), datetime(2026, 10, 27, 16, 3, 4, tzinfo=timezone.utc)
        )
        text = items[0]["parts"][0]["text"]
        self.assertIn("2026-10-28 00:03:04，星期三", text)
        self.assertIn("Asia/Shanghai", text)

    def test_refresh_is_idempotent_and_keeps_persona(self):
        first = candidate.inject_runtime_clock(
            serialize_prompt_items(fixture()), datetime(2026, 10, 27, 16, 3, 4, tzinfo=timezone.utc)
        )
        second = candidate.inject_runtime_clock(first, datetime(2026, 10, 27, 16, 4, 5, tzinfo=timezone.utc))
        text = second[0]["parts"][0]["text"]
        self.assertIn("人格和工具约定", text)
        self.assertIn("2026-10-28 00:04:05", text)
        self.assertNotIn("2026-10-28 00:03:04", text)
        self.assertEqual(text.count("\n<runtime_clock>\n"), 1)
        self.assertEqual(len(first), len(second))

    def test_only_current_tail_clock_removed(self):
        original = serialize_prompt_items(fixture())
        items = candidate.inject_runtime_clock(original)
        bodies = [p.get("text") for item in items for p in item.get("parts", [])]
        self.assertIn("时间：2026-10-27 00:00:00", bodies)
        self.assertNotIn("时间：2026-10-28 14:00:00", bodies)
        self.assertIn("[msg_id:test]真实聊天", bodies)
        self.assertEqual(len(original), 5)  # input remains unchanged

    def test_native_context_roundtrip(self):
        original = fixture()
        payload = candidate.inject_runtime_clock(serialize_prompt_items(original))
        restored = deserialize_prompt_items(
            payload, item_schema_version=1, mode=ContextProtocolMode.REQUEST_CONTEXT, original_items=original
        )
        self.assertEqual(restored[0].role, RoleType.System)
        self.assertIn("<runtime_clock>", restored[0].parts[0].text)

    def test_hook_keeps_direct_reply_schema(self):
        from src.maisaka.builtin_tool.reply import get_tool_spec

        plugin = candidate.create_plugin()
        result = asyncio.run(
            plugin.configure_reply_handoff(
                items=serialize_prompt_items(fixture()), tool_definitions=[get_tool_spec().to_llm_definition()]
            )
        )
        changes = result["modified_kwargs"]
        self.assertIn("<runtime_clock>", changes["items"][0]["parts"][0]["text"])
        definition = changes["tool_definitions"][0]
        function = definition.get("function", definition)
        self.assertEqual(function.get("parameters_schema", function.get("parameters"))["required"], ["msg_id", "text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
