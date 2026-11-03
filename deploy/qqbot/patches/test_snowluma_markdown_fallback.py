"""Run against an installed SnowLuma adapter; no platform connection is created.

SNOWLUMA_PLUGIN_DIR=/path/to/MaiBot-SnowLuma-Adapter python test_snowluma_markdown_fallback.py
"""

from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import ModuleType
from unittest import IsolatedAsyncioTestCase

import os
import sys
import unittest


class MarkdownFallbackTests(IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(os.environ["SNOWLUMA_PLUGIN_DIR"]).resolve()
        package = ModuleType("snowluma_replay")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        module = import_module("snowluma_replay.codecs.inbound.message_codec")
        self.codec = module.QQInboundCodec(None, None, None, lambda: None)

    async def convert(self, segments):
        payload = {"message": segments}
        before = deepcopy(payload)
        result, mentioned = await self.codec.convert_segments(payload, "bot")
        self.assertEqual(payload, before)
        self.assertFalse(mentioned)
        return result

    async def test_blank_line_difference_is_one_markdown_body(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "看到了。\n\n需要什么直接说。"}},
            {"type": "text", "data": {"text": "看到了。\n需要什么直接说。"}},
        ])
        self.assertEqual(result, [{"type": "text", "data": "看到了。\n\n需要什么直接说。"}])

    async def test_exact_fallback_is_one_body(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "原始正文"}},
            {"type": "text", "data": {"text": "原始正文"}},
        ])
        self.assertEqual(result, [{"type": "text", "data": "原始正文"}])

    async def test_line_endings_and_spacing(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "First line\r\n\r\nSecond line"}},
            {"type": "text", "data": {"text": "First  line\nSecond line\n"}},
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["data"], "First line\r\n\r\nSecond line")

    async def test_normal_repeated_text_is_preserved(self):
        segments = [{"type": "text", "data": {"text": "收到"}}] * 2
        self.assertEqual(await self.convert(segments), [{"type": "text", "data": "收到"}] * 2)

    async def test_repetition_inside_plain_text_is_preserved(self):
        result = await self.convert([{"type": "text", "data": {"text": "收到 收到"}}])
        self.assertEqual(result, [{"type": "text", "data": "收到 收到"}])

    async def test_distinct_text_after_markdown_is_preserved(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "正文"}},
            {"type": "text", "data": {"text": "补充说明"}},
        ])
        self.assertEqual(result, [{"type": "text", "data": "正文"}, {"type": "text", "data": "补充说明"}])

    async def test_word_boundaries_are_preserved(self):
        for markdown, text in [("a b", "ab"), ("中 文", "中文")]:
            with self.subTest(markdown=markdown):
                result = await self.convert([
                    {"type": "markdown", "data": {"content": markdown}},
                    {"type": "text", "data": {"text": text}},
                ])
                self.assertEqual(len(result), 2)

    async def test_repeated_markdown_is_preserved(self):
        result = await self.convert([{"type": "markdown", "data": {"content": "正文"}}] * 2)
        self.assertEqual(result, [{"type": "text", "data": "正文"}] * 2)

    async def test_nonadjacent_text_is_preserved(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "正文"}},
            {"type": "text", "data": {"text": "另外一句"}},
            {"type": "text", "data": {"text": "正文"}},
        ])
        self.assertEqual(len(result), 3)

    async def test_multiple_compatibility_pairs(self):
        result = await self.convert([
            {"type": "markdown", "data": {"content": "第一段\n\n补充"}},
            {"type": "text", "data": {"text": "第一段\n补充"}},
            {"type": "markdown", "data": {"content": "第二段"}},
            {"type": "text", "data": {"text": "第二段"}},
        ])
        self.assertEqual(result, [{"type": "text", "data": "第一段\n\n补充"}, {"type": "text", "data": "第二段"}])


if __name__ == "__main__":
    unittest.main()
