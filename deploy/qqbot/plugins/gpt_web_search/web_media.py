"""Download a public image before passing bytes to the QQ image sender."""

import asyncio
from io import BytesIO
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit
import warnings

import aiohttp
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


def public_address(value):
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


def validate_url(url):
    if not isinstance(url, str) or not url.strip() or len(url) > 8192:
        raise ValueError("需要实际图片的完整 HTTP(S) 地址")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("图片地址必须是无凭据的 HTTP(S) URL")
    _ = parts.port
    try:
        ipaddress.ip_address(parts.hostname)
    except ValueError:
        pass
    else:
        if not public_address(parts.hostname):
            raise ValueError("只支持下载互联网图片，不能使用内网或本机地址")
    return parts


class PublicResolver(aiohttp.resolver.DefaultResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        answers = await super().resolve(host, port, family)
        if not answers or any(not public_address(answer["host"]) for answer in answers):
            raise ValueError("图片主机没有可用的公网地址")
        # These exact validated addresses are handed to the connector.
        return answers


def verify_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("JPEG", "PNG", "GIF", "WEBP", "BMP", "AVIF"):
                    raise ValueError("图片格式不支持，请换 JPEG、PNG、GIF 或 WebP 图片")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("图片分辨率过大，请换较小的版本")
                image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("下载结果不是有效图片，可能是网页或防盗链页面；请换图片直链") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("图片分辨率过大，请换较小的版本") from exc
    return data


async def download_web_image(url):
    current = url.strip() if isinstance(url, str) else url
    connector = aiohttp.TCPConnector(resolver=PublicResolver())
    async with (
        asyncio.timeout(30),
        aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=25, sock_connect=10),
            cookie_jar=aiohttp.DummyCookieJar(),
            trust_env=False,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "image/avif,image/webp,image/*;q=0.9,*/*;q=0.5"},
        ) as session,
    ):
        for _ in range(6):
            parts = validate_url(current)
            async with session.get(current, allow_redirects=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("图片重定向缺少目标地址")
                    target = urljoin(current, location)
                    next_parts = validate_url(target)
                    if parts.scheme == "https" and next_parts.scheme != "https":
                        raise ValueError("图片地址跳转到不安全的 HTTP 链接，请换来源")
                    current = target
                    continue
                if response.status != 200:
                    raise ValueError(f"图片站返回 HTTP {response.status}，可能有防盗链；请换搜索结果或提供来源页链接")
                if response.content_length and response.content_length > MAX_IMAGE_BYTES:
                    raise ValueError("图片超过 15 MiB，请换较小的版本")
                data = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError("图片超过 15 MiB，请换较小的版本")
                return await asyncio.to_thread(verify_image, bytes(data))
        raise ValueError("图片重定向次数过多，请换图片直链")
