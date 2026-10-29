"""Use real plugin methods and mocked QQ transport; never send group messages."""

import base64
import importlib
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

plugin_module = importlib.import_module("deploy.qqbot.plugins.gpt_web_search.plugin")


def sample_image():
    buf = BytesIO()
    Image.new("RGB", (20, 10), "blue").save(buf, format="PNG")
    return buf.getvalue()


class ImageToolTests(unittest.IsolatedAsyncioTestCase):
    def plugin(self, result):
        plugin = plugin_module.GPTWebSearchPlugin()
        sender = AsyncMock(return_value=result)
        plugin._ctx = SimpleNamespace(send=SimpleNamespace(image=sender))
        return plugin, sender

    async def test_search_result_can_reach_image_sender(self):
        plugin, sender = self.plugin({"success": True, "sent": True, "message_id": "test-id"})
        self.assertTrue(hasattr(plugin, "send_web_image"), "Search has image URLs but no tool can send an actual image")
        with patch.object(plugin_module, "download_web_image", AsyncMock(return_value=sample_image())):
            result = await plugin.send_web_image(image_url="https://example.org/cat.png", stream_id="test-session")
        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "test-id")
        args, kwargs = sender.call_args
        self.assertEqual(base64.b64decode(args[0]), sample_image())
        self.assertEqual(args[1], "test-session")
        self.assertTrue(kwargs["return_details"])

    async def test_platform_rejection_is_not_success(self):
        plugin, sender = self.plugin({"success": False, "sent": False, "error": "test rejection"})
        with patch.object(plugin_module, "download_web_image", AsyncMock(return_value=sample_image())):
            result = await plugin.send_web_image(image_url="https://example.org/cat.png", stream_id="test-session")
        self.assertFalse(result["success"])
        self.assertEqual(sender.await_count, 1)

    async def test_non_image_download_never_reaches_qq(self):
        plugin, sender = self.plugin({"success": True, "sent": True})
        with patch.object(
            plugin_module, "download_web_image", AsyncMock(side_effect=ValueError("URL points to HTML, not an image"))
        ):
            result = await plugin.send_web_image(image_url="https://example.org/page", stream_id="test-session")
        self.assertFalse(result["success"])
        sender.assert_not_awaited()

    async def test_no_session_never_sends(self):
        plugin, sender = self.plugin({"success": True, "sent": True})
        result = await plugin.send_web_image(image_url="https://example.org/cat.png")
        self.assertFalse(result["success"])
        sender.assert_not_awaited()


class DownloadValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.media = importlib.import_module(plugin_module.__package__ + ".web_media")

    def test_html_is_not_an_image(self):
        with self.assertRaises(ValueError):
            self.media.verify_image(b"<html>Access denied</html>")
        self.assertEqual(self.media.verify_image(sample_image()), sample_image())

    def test_private_addresses_and_unsupported_schemes(self):
        for url in (
            "http://127.0.0.1/image",
            "http://192.168.2.238/image",
            "http://[::1]/image",
            "file:///etc/passwd",
            "https://user:pass@example.org/image",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.media.validate_url(url)

    async def test_hostname_resolving_to_private_ip_is_blocked(self):
        import aiohttp

        resolver = self.media.PublicResolver()
        with patch.object(aiohttp.resolver.DefaultResolver, "resolve", AsyncMock(return_value=[{"host": "127.0.0.1"}])):
            with self.assertRaises(ValueError):
                await resolver.resolve("example.org", 443)
        await resolver.close()


if __name__ == "__main__":
    unittest.main()
