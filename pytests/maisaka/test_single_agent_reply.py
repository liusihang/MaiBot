"""Exercise native reply processing with transport intercepted and generation forbidden."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from jsonschema import validate
from src.common.data_models.message_component_data_model import AtComponent, ImageComponent
from src.common.data_models.reply_generation_data_models import LLMCompletionResult, ReplyGenerationResult
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import reply
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext


class DirectReplyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.target = SimpleNamespace(
            message_id="test-target",
            message_info=SimpleNamespace(
                user_info=SimpleNamespace(user_id="test-user", user_cardname="测试群友", user_nickname="测试群友")
            ),
        )
        self.runtime = SimpleNamespace(
            session_id="single-agent-acceptance",
            log_prefix="[single-agent-acceptance]",
            _chat_history=[],
            chat_stream=SimpleNamespace(
                platform="qq", account_id="test-bot", is_group_session=True, group_id="test-group", user_id=None
            ),
            find_source_message_by_id=lambda key: self.target if key == "test-target" else None,
            record_planner_reply=Mock(),
            _update_stage_status=Mock(),
        )
        self.ctx = BuiltinToolRuntimeContext(SimpleNamespace(), self.runtime)

        async def hook(name, **kwargs):
            return SimpleNamespace(kwargs=kwargs)

        self.ctx.get_runtime_manager = lambda: SimpleNamespace(invoke_hook=hook)
        self.ctx._resolve_at_attachment = lambda key: AtComponent(target_user_id="test-user")
        image = object.__new__(ImageComponent)
        image.content = "测试图片"
        image.binary_data = b"test"
        image.binary_hash = "test-only"
        self.ctx._resolve_image_attachment = AsyncMock(return_value=image)
        self.sent = []

        async def send(**kwargs):
            self.sent.append(kwargs)
            return SimpleNamespace(message_id=f"intercepted-{len(self.sent)}")

        self.send = AsyncMock(side_effect=send)
        self.patches = [
            patch.object(reply, "find_messages", return_value=[]),
            # Assert direct-body identity without the independently configured random typo effect.
            patch.object(reply.config_module.global_config.chinese_typo, "enable", False),
            patch.object(reply.send_service, "_send_to_target_with_message", self.send),
            patch.object(
                reply.replyer_manager, "get_replyer", side_effect=AssertionError("Replyer generation forbidden")
            ),
            patch.object(reply, "_resolve_reply_quote", AsyncMock(return_value=(True, 0.5))),
            patch.object(reply, "build_reply_extensions_schema", return_value=None),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()

    async def invoke(self, **extra):
        args = {"msg_id": "test-target", "text": "这句是主Agent写好的。下一句也不改写。", **extra}
        return await reply.handle_tool(self.ctx, ToolInvocation(call_id="test-call", tool_name="reply", arguments=args))

    async def test_direct_body_never_calls_replyer(self):
        result = await self.invoke()
        self.assertTrue(result.success, result.error_message)
        self.assertEqual([x["processed_plain_text"] for x in self.sent], ["这句是主Agent写好的", "下一句也不改写"])
        self.assertEqual(result.metadata["reply_mode"], "single_agent")
        self.assertEqual(result.metadata["replyer_llm_calls"], 0)
        self.assertEqual(result.metadata["monitor_detail"]["generation_attempts"], [])
        reply.replyer_manager.get_replyer.assert_not_called()
        self.assertTrue(self.sent[0]["set_reply"])
        self.assertIs(self.sent[0]["reply_message"], self.target)
        self.assertTrue(self.sent[0]["sync_to_maisaka_history"])

    async def test_reply_resolves_a_persisted_target_outside_the_runtime_mirror(self):
        self.runtime.find_source_message_by_id = lambda key: None
        reply.find_messages.return_value = [self.target]
        result = await self.invoke(text="保留的旧消息也能回复。")
        self.assertTrue(result.success, result.error_message)
        reply.find_messages.assert_called_once_with(
            session_id="single-agent-acceptance", message_id="test-target", limit=1
        )
        self.assertIs(self.sent[0]["reply_message"], self.target)

    async def test_invalid_target_is_a_known_pre_dispatch_failure(self):
        result = await self.invoke(msg_id="not-in-this-chat")
        self.assertFalse(result.success)
        self.assertEqual(result.metadata["execution_outcome"], "not_executed")
        self.assertFalse(self.sent)

    async def test_schema_and_options(self):
        schema = reply.get_tool_spec().parameters_schema
        validate({"msg_id": "test-target", "text": "正文"}, schema)
        self.assertEqual(schema["required"], ["msg_id", "text"])
        self.assertNotIn("reply_reference", schema["properties"])
        self.assertNotIn("expression_intent", schema["properties"])
        self.assertIn("attach_pic", schema["properties"])

    async def test_picture_and_at_attachments_preserved(self):
        result = await self.invoke(
            text="图在这。", attach_at=["test-target"], attach_pic=[{"msg_id": "test-target", "index": 0}]
        )
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(len(self.sent), 2)
        self.assertIsInstance(self.sent[0]["message_sequence"].components[0], AtComponent)
        self.assertIsInstance(self.sent[1]["message_sequence"].components[0], ImageComponent)
        self.ctx._resolve_image_attachment.assert_awaited_once()
        reply.replyer_manager.get_replyer.assert_not_called()

    async def test_empty_text_rejected(self):
        result = await self.invoke(text="  ")
        self.assertFalse(result.success)
        self.send.assert_not_called()
        reply.replyer_manager.get_replyer.assert_not_called()

    async def test_reference_is_not_an_agent_fallback(self):
        result = await reply.handle_tool(
            self.ctx,
            ToolInvocation(
                call_id="legacy-agent",
                tool_name="reply",
                arguments={"msg_id": "test-target", "reply_reference": "请另一个模型写正文"},
            ),
        )
        self.assertFalse(result.success)
        self.assertIn("text", result.error_message)
        reply.replyer_manager.get_replyer.assert_not_called()

    async def test_missing_target_rejected(self):
        result = await self.invoke(msg_id="missing")
        self.assertFalse(result.success)
        self.send.assert_not_called()

    async def test_partial_failure_preserves_receipts(self):
        self.send.side_effect = [SimpleNamespace(message_id="intercepted-first"), None]
        result = await self.invoke()
        self.assertFalse(result.success)
        self.assertEqual(result.metadata["sent_message_ids"], ["intercepted-first"])
        self.assertEqual([x["success"] for x in result.metadata["send_results"]], [True, False])
        self.assertEqual(self.send.await_count, 2)
        reply.replyer_manager.get_replyer.assert_not_called()

    async def test_send_exception_preserves_receipts(self):
        self.send.side_effect = [
            SimpleNamespace(message_id="intercepted-first"),
            RuntimeError("simulated transport error"),
        ]
        result = await self.invoke()
        self.assertFalse(result.success)
        self.assertEqual(result.metadata["sent_message_ids"], ["intercepted-first"])
        self.assertEqual(self.send.await_count, 2)

    async def test_wrapped_arguments(self):
        result = await reply.handle_tool(
            self.ctx,
            ToolInvocation(
                call_id="wrapped",
                tool_name="reply",
                arguments={"arguments": json.dumps({"msg_id": "test-target", "text": "直接发这句"})},
            ),
        )
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(self.sent[0]["processed_plain_text"], "直接发这句")

    async def test_explicit_plugin_generation_api_remains(self):
        generated = ReplyGenerationResult(
            success=True, completion=LLMCompletionResult(response_text="插件明确请求生成的正文")
        )
        generator = SimpleNamespace(generate_reply_with_context=AsyncMock(return_value=(True, generated)))
        reply.replyer_manager.get_replyer.side_effect = None
        reply.replyer_manager.get_replyer.return_value = generator
        result = await reply.execute_reply(
            self.ctx, {"msg_id": "test-target", "reply_reference": "插件生成请求"}, call_id="plugin-api"
        )
        self.assertTrue(result.success, result.error_message)
        generator.generate_reply_with_context.assert_awaited_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
