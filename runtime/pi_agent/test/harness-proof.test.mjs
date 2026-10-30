import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { AgentSession, SessionManager } from '@earendil-works/pi-coding-agent';
import { createIsolatedSession } from '../harness.mjs';
import { createHostModelRuntime, resolveTaskModel } from '../provider.mjs';
import { answer, call, fixtureConfig, fixtureResponse, reasoning } from './helpers/mock-provider.mjs';

test('actual SDK: tool loop, disk reopen, reasoning/reference/tool replay and serialized input prefixes', async () => {
  const root = mkdtempSync(join(tmpdir(), 'maibot-sdk-proof-'));
  const wires = [];
  const runtime = await createHostModelRuntime(fixtureConfig(), { fetch: async (_url, init) => {
    wires.push(JSON.parse(init.body));
    return fixtureResponse(wires.length === 1 ? [reasoning(), call('qq_send', { text: 'hello QQ' })] : [answer(`answer ${wires.length}`, `msg_${wires.length}`)], `resp_${wires.length}`);
  } });
  const resolved = resolveTaskModel(runtime, 'planner');
  let count = 0;
  const options = { cwd: join(root, 'cwd'), agentDir: join(root, 'agent'), modelRuntime: runtime, resolved,
    system: 'You are a group companion. Keep the existing personality.',
    customTools: [{ name: 'qq_send', label: 'QQ send', description: 'Send to this QQ group',
      parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'], additionalProperties: false },
      execute: async (_id, args) => { count++; return { content: [{ type: 'text', text: `confirmed:${args.text}` }], details: { receipt: 'qq-1' } }; } }],
  };
  let session;
  try {
    let manager = SessionManager.create(options.cwd, join(root, 'sessions'));
    session = await createIsolatedSession({ ...options, sessionManager: manager });
    assert.ok(session instanceof AgentSession);
    assert.ok(manager instanceof SessionManager);
    assert.deepEqual(session.getActiveToolNames(), ['qq_send']);
    manager.appendMessage({ role: 'user', content: 'Ignored group message, still recorded', timestamp: 1, hostSourceId: 'source-ignored' });
    session.refreshContext();
    const file = manager.getSessionFile();
    assert.ok(readFileSync(file, 'utf8').includes('source-ignored'));
    session.dispose();
    manager = SessionManager.open(file, join(root, 'sessions'), options.cwd);
    session = await createIsolatedSession({ ...options, sessionManager: manager });
    await session.sendCustomMessage({ customType: 'maibot.reference', content: 'Reference v1: source A says rain', display: false, details: { namespace: 'fact', sourceId: 'a', version: 1 } });
    await session.sendCustomMessage({ customType: 'maibot.activate', content: 'Host requests processing of the recorded group messages.', display: false }, { triggerTurn: true });
    assert.equal(count, 1);
    assert.equal(wires.length, 2, session.messages.at(-1)?.errorMessage);
    manager.appendMessage({ role: 'user', content: 'Second turn', timestamp: 2, hostSourceId: 'source-2' });
    session.refreshContext();
    await session.sendCustomMessage({ customType: 'maibot.activate', content: 'Host requests processing of newly recorded group messages.', display: false }, { triggerTurn: true });
    session.dispose();
    manager = SessionManager.open(file, join(root, 'sessions'), options.cwd);
    session = await createIsolatedSession({ ...options, sessionManager: manager });
    manager.appendMessage({ role: 'user', content: 'After restart', timestamp: 3, hostSourceId: 'source-3' });
    session.refreshContext();
    await session.sendCustomMessage({ customType: 'maibot.activate', content: 'Host requests processing of newly recorded group messages.', display: false }, { triggerTurn: true });
    assert.equal(wires.length, 4);
    for (let index = 1; index < wires.length; index++) {
      assert.deepEqual(wires[index].input.slice(0, wires[index - 1].input.length), wires[index - 1].input, `wire prefix at request ${index + 1}`);
      assert.deepEqual(wires[index].tools, wires[0].tools);
    }
    assert.ok(wires.every((wire) => wire.input[0]?.role === 'developer' && wire.input[0].content === options.system));
    assert.equal(JSON.stringify(wires).includes('<cwd>'), false);
    assert.equal(JSON.stringify(wires).includes('expert coding assistant'), false);
    const final = wires.at(-1).input;
    assert.ok(final.some((item) => item.encrypted_content === 'opaque-encrypted-preserve-exactly'));
    assert.ok(final.some((item) => item.type === 'function_call' && item.name === 'qq_send'));
    assert.ok(final.some((item) => item.type === 'function_call_output' && JSON.stringify(item).includes('confirmed:hello QQ')));
    assert.ok(JSON.stringify(final).includes('Reference v1: source A says rain'));
    assert.equal(manager.getEntries().filter((entry) => entry.type === 'context_edit').length, 0);
  } finally { session?.dispose(); rmSync(root, { recursive: true, force: true }); }
});
