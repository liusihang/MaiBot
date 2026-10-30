from pathlib import Path

import asyncio
import sys

import pytest

from src.maisaka.pi_runtime.bridge import PiBridge, PiBridgeError


@pytest.mark.asyncio
async def test_callback_ack_before_result_and_interleaved_requests(tmp_path: Path):
    script = tmp_path / "sidecar.py"
    script.write_text("""import sys,json
for line in sys.stdin:
 p=json.loads(line)
 if p.get("method") == "echo":
  print(json.dumps({"id":"cb", "method":"event", "params":{"event":"durable"}}),flush=True)
  ack=json.loads(sys.stdin.readline())
  print(json.dumps({"id":p["id"],"result":ack["result"]}),flush=True)
""")
    seen = []

    async def callback(method, params):
        await asyncio.sleep(0)
        seen.append((method, params))
        return {"persisted": True}

    bridge = PiBridge([sys.executable, str(script)], callback)
    try:
        assert await bridge.request("echo") == {"persisted": True}
        assert seen == [("event", {"event": "durable"})]
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_crash_fails_waiting_request(tmp_path: Path):
    async def callback(method, params):
        raise AssertionError("not expected")

    bridge = PiBridge([sys.executable, "-c", "import sys; sys.stdin.readline()"], callback)
    try:
        with pytest.raises(PiBridgeError, match="closed"):
            await asyncio.wait_for(bridge.request("prompt"), 5)
    finally:
        await bridge.close()
