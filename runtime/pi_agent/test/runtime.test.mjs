import assert from 'node:assert/strict';
import { appendFileSync, fsyncSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { AgentSession, SessionManager } from '@earendil-works/pi-coding-agent';
import { PiRuntime } from '../runtime.mjs';
import { createHostModelRuntime } from '../provider.mjs';
import { answer, call, fixtureConfig, fixtureResponse, reasoning } from './helpers/mock-provider.mjs';

const user = (text, id = text) => ({ role: 'user', content: text, timestamp: 1, hostEventId: id });
const tool = (name) => ({ name, description: `Business ${name}`, parameters: { type: 'object', additionalProperties: true } });
const success = (text = 'ok', extra = {}) => ({ content: [{ type: 'text', text }], details: {}, ...extra });
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const ref = (version = 1, content = 'source details') => ({ namespace: 'fact', sourceId: 'source-a', version, content });
const rows = (owned) => readFileSync(owned.manager.getSessionFile(), 'utf8').trim().split('\n').map(JSON.parse);
const strings = (state) => state.messages.filter((m) => m.role === 'user').map((m) => m.content);

function harness(t, { response, execute = async () => success(), event = async () => {}, authorize = async () => ({ allowed: true }), beforePersist, summarize, syncFile, root } = {}) {
  const dir = root ?? mkdtempSync(join(tmpdir(), 'maibot-runtime-test-'));
  const wires = [], events = [], calls = [];
  const runtime = new PiRuntime(async (method, params, signal) => {
    if (method === 'authorize') return authorize(params, signal);
    if (method === 'tool') { calls.push(structuredClone(params)); return execute(params, signal, runtime); }
    if (method === 'event') { events.push(structuredClone(params)); return event(params, runtime); }
    assert.fail(`Forbidden callback: ${method}`);
  }, { createRuntime: (config) => createHostModelRuntime(config, { fetch: async (_url, init) => {
    const wire = JSON.parse(init.body); wires.push(wire);
    const result = await response?.(wires.length, wire, init.signal, runtime);
    return result instanceof Response ? result : fixtureResponse(result ?? [answer('done', `m${wires.length}`)], `r${wires.length}`);
  } }), beforePersist, summarize, syncFile });
  t.after(async () => { await runtime.shutdown(); if (!root) rmSync(dir, { recursive: true, force: true }); });
  const initialize = (id = 'group-a', tools = [], extra = {}) => runtime.handle('init', {
    sessionId: id, sessionDir: dir, system: 'Fixed group persona and political constraints', tools,
    toolBindings: Object.fromEntries(tools.map((entry) => [entry.name, { provider: 'host' }])),
    config: fixtureConfig(), taskName: 'planner', ...extra,
  });
  // Native replay assertions read the local SDK manager, never a production RPC snapshot.
  const inspect = (state) => {
    const owned = runtime.sessions.get(state?.sessionId);
    return !owned || state.closed ? state : { ...state,
      messages: structuredClone(owned.manager.buildSessionProjection().messages),
      references: structuredClone(owned.visibleReferences()),
      queuedMessages: structuredClone(owned.pending.filter((entry) => entry.kind === 'input').map((entry) => entry.value)),
      archive: { ...state.archive, eventIds: owned.manager.getEntries().map((entry) => entry.id) } };
  };
  const handle = async (method, params) => inspect(await runtime.handle(method, params));
  const prompt = (text = 'hello', id = 'group-a', extra = {}) => handle('prompt', { sessionId: id, additions: [user(text)], ...extra });
  const state = (id = 'group-a') => handle('state', { sessionId: id });
  return { runtime, initialize, prompt, state, handle, wires, events, calls, dir };
}

function prefixes(wires) {
  for (let i = 1; i < wires.length; i++) {
    assert.deepEqual(wires[i].input.slice(0, wires[i - 1].input.length), wires[i - 1].input);
    assert.deepEqual(wires[i].tools, wires[0].tools);
  }
}

test('official AgentSession search → fetch → direct reply preserves native wire prefix and fixed tools across restart', async (t) => {
  const h = harness(t, { response: (n) => n < 4 ? [reasoning(), call(['search', 'fetch', 'reply'][n - 1], { text: 'exact main text' }, `c${n}`, `fc${n}`)] : undefined,
    execute: ({ name }) => success(name, { terminate: name === 'reply' }) });
  await h.initialize('group-a', ['search', 'fetch', 'reply'].map(tool));
  const owned = h.runtime.sessions.get('group-a');
  assert.ok(owned.session instanceof AgentSession);
  assert.ok(owned.manager instanceof SessionManager);
  assert.deepEqual(owned.session.getActiveToolNames(), ['search', 'fetch', 'reply']);
  await h.prompt('first', 'group-a', { references: [ref()] });
  assert.deepEqual(h.calls.map((entry) => entry.name), ['search', 'fetch', 'reply']);
  assert.equal(h.calls.at(-1).arguments.text, 'exact main text');
  assert.equal(h.wires.length, 3);
  const before = await h.state();
  await h.handle('close', { sessionId: 'group-a' });
  await h.initialize('group-a', ['search', 'fetch', 'reply'].map(tool));
  assert.deepEqual(JSON.parse(JSON.stringify((await h.state()).messages)), JSON.parse(JSON.stringify(before.messages)));
  await h.prompt('second');
  prefixes(h.wires);
  assert.ok(h.wires.every((wire) => wire.input[0]?.role === 'developer' && wire.input[0].content === 'Fixed group persona and political constraints'));
  assert.equal(JSON.stringify(h.wires).includes('<cwd>'), false);
  assert.equal(JSON.stringify(h.wires).includes('expert coding assistant'), false);
  assert.ok(JSON.stringify(h.wires.at(-1)).includes('opaque-encrypted-preserve-exactly'));
  assert.ok(JSON.stringify(h.wires.at(-1)).includes('source details'));
  assert.equal(h.runtime.sessions.get('group-a').manager.getEntries().filter((entry) => entry.type === 'context_edit').length, 0);
});

test('ignored input durable ACK precedes any inference; retransmission uses whole archive and conflicting content rejects', async (t) => {
  const h = harness(t);
  await h.initialize();
  const first = await h.handle('append', { sessionId: 'group-a', additions: [user('ignored', 'stable')] });
  assert.deepEqual(first.acceptedSourceIds, ['stable']);
  const owned = h.runtime.sessions.get('group-a');
  assert.ok(rows(owned).some((row) => row.message?.hostEventId === 'stable'));
  assert.equal(h.wires.length, 0);
  await h.handle('close', { sessionId: 'group-a' });
  await h.initialize();
  await h.handle('append', { sessionId: 'group-a', additions: [user('ignored', 'stable')] });
  await assert.rejects(h.handle('append', { sessionId: 'group-a', additions: [user('changed', 'stable')] }), { code: 'SOURCE_CONFLICT' });
  await h.prompt('activate');
  assert.equal(JSON.stringify(h.wires[0]).match(/ignored/g)?.length, 1);
});

test('groups execute independently with no context, tools or archive access leakage', async (t) => {
  const startedA = deferred(), releaseA = deferred();
  const h = harness(t, { response: async (_n, wire) => {
    if (JSON.stringify(wire).includes('secret-a')) { startedA.resolve(); await releaseA.promise; }
  } });
  await h.initialize('a', [tool('only_a')]); await h.initialize('b', [tool('only_b')]);
  const a = h.prompt('secret-a', 'a'); await startedA.promise;
  const b = await h.prompt('secret-b', 'b');
  assert.equal((await h.state('a')).isRunning, true);
  releaseA.resolve(); const doneA = await a;
  assert.equal(JSON.stringify(doneA).includes('secret-b'), false);
  assert.equal(JSON.stringify(b).includes('secret-a'), false);
  await assert.rejects(h.handle('read_archive', { sessionId: 'b', eventIds: doneA.archive.eventIds.slice(-1) }), { code: 'ARCHIVE_NOT_FOUND' });
});

test('durable assistant anchors and tool results precede every sequential side effect', async (t) => {
  const h = harness(t, { response: (n) => n === 1 ? [call('first', {}, 'a', 'fa'), call('second', {}, 'b', 'fb')] : undefined,
    execute: ({ toolCallId, messageId }, _signal, runtime) => {
      const disk = rows(runtime.sessions.get('group-a'));
      assert.ok(disk.some((entry) => entry.id === messageId && entry.message.role === 'assistant'));
      if (toolCallId === 'b|fb') assert.ok(disk.some((entry) => entry.message?.toolCallId === 'a|fa'));
      return success();
    } });
  await h.initialize('group-a', [tool('first'), tool('second')]);
  await h.prompt();
  assert.deepEqual(h.calls.map((entry) => entry.toolCallId), ['a|fa', 'b|fb']);
});

test('mid-tool and reentrant arrivals get durable pending entries then appear once after the complete tool batch', async (t) => {
  const started = deferred(), finish = deferred(); let reentrant = false;
  const h = harness(t, { response: (n) => n === 1 ? [call('fetch', {}, 'a', 'fa')] : undefined,
    execute: async () => { started.resolve(); await finish.promise; return success('document'); },
    event: async ({ event }, runtime) => {
      if (!reentrant && event.type === 'message_end' && event.message.role === 'toolResult') {
        reentrant = true; await runtime.handle('append', { sessionId: 'group-a', additions: [user('reentrant')] });
      }
    } });
  await h.initialize('group-a', [tool('fetch')]);
  const running = h.prompt(); await started.promise;
  const queued = await h.prompt('mid-tool');
  assert.equal(queued.queued, true);
  assert.ok(rows(h.runtime.sessions.get('group-a')).some((row) => row.customType === 'maibot.inbox' && row.data.message.content === 'mid-tool'));
  finish.resolve(); const final = await running;
  assert.deepEqual(strings(final), ['hello', 'mid-tool', 'reentrant']);
  assert.deepEqual(final.queuedMessages, []);
  assert.ok(JSON.stringify(h.wires[1]).includes('reentrant'));
  prefixes(h.wires);
});

test('wait/stop terminates without extra request, preserves pending FIFO and blocks remaining side effects', async (t) => {
  const h = harness(t, { response: (n) => n === 1 ? [call('wait', {}, 'a', 'fa'), call('send', {}, 'b', 'fb')] : undefined,
    execute: async (_params, _signal, runtime) => {
      await runtime.handle('append', { sessionId: 'group-a', additions: [user('older queued')] });
      return success('waiting', { terminate: true });
    } });
  await h.initialize('group-a', [tool('wait'), tool('send')]);
  const waiting = await h.prompt('original');
  assert.equal(h.wires.length, 1); assert.equal(h.calls.length, 1);
  assert.equal(waiting.messages.find((m) => m.toolCallId === 'b|fb').isError, true);
  assert.equal(waiting.queuedMessages.length, 1);
  const resumed = await h.prompt('newer');
  assert.deepEqual(strings(resumed), ['original', 'older queued', 'newer']);
  prefixes(h.wires);
});

test('known failed tool result persists and continues; unknown effect pauses and receipt recovery never re-executes', async (t) => {
  const h = harness(t, { response: (n) => n === 1 ? [call('send', {}, 'a', 'fa')] : undefined,
    execute: async () => { throw new Error('acknowledgement lost after side effect'); } });
  await h.initialize('group-a', [tool('send')]);
  const paused = await h.prompt();
  assert.equal(h.wires.length, 1); assert.equal(paused.paused, true);
  assert.equal(paused.messages.find((m) => m.toolCallId === 'a|fa').isError, true);
  await assert.rejects(h.prompt('must wait'), { code: 'UNRESOLVED_TOOL_CALLS' });
  await h.handle('close', { sessionId: 'group-a' }); await h.initialize('group-a', [tool('send')]);
  assert.equal((await h.state()).paused, true);
  await h.handle('resolve_tools', { sessionId: 'group-a', results: [{ role: 'toolResult', toolCallId: 'a|fa', toolName: 'send',
    content: [{ type: 'text', text: 'confirmed durable receipt' }], isError: false, timestamp: 2 }] });
  await h.prompt('resume');
  assert.equal(h.calls.length, 1);
  assert.equal((await h.state()).messages.filter((m) => m.role === 'toolResult' && m.toolCallId === 'a|fa').length, 1);
  assert.ok(JSON.stringify(h.wires[1]).includes('confirmed durable receipt'));
  prefixes(h.wires);
  const known = harness(t, { response: (n) => n === 1 ? [call('search', {})] : undefined,
    execute: () => success('not found', { isError: true }) });
  await known.initialize('group-a', [tool('search')]);
  assert.equal((await known.prompt()).paused, false); assert.equal(known.wires.length, 2);
});

test('cancellation during effect preserves unknown outcome and does not authorize another provider request', async (t) => {
  const started = deferred();
  const h = harness(t, { response: () => [call('send', {})], execute: async (_params, signal) => {
    started.resolve(); await new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true }));
  } });
  await h.initialize('group-a', [tool('send')]);
  const running = h.prompt(); await started.promise;
  await h.handle('abort', { sessionId: 'group-a' });
  const result = await running;
  assert.equal(result.paused, true); assert.equal(h.wires.length, 1);
  assert.ok(result.messages.some((m) => m.role === 'toolResult' && m.isError));
});

test('abort while authorize is pending sends no model request and preserves source input', async (t) => {
  const started = deferred(), release = deferred();
  const h = harness(t, { authorize: async () => { started.resolve(); await release.promise; return { allowed: true }; } });
  await h.initialize(); const running = h.prompt(); await started.promise;
  const abort = h.handle('abort', { sessionId: 'group-a' }); release.resolve();
  await abort; const result = await running;
  assert.equal(h.wires.length, 0); assert.deepEqual(strings(result), ['hello']);
});

test('reference dedup is visible-epoch scoped, version changes append, whole-archive input dedup survives compaction', async (t) => {
  const h = harness(t); await h.initialize();
  await h.prompt('first', 'group-a', { references: [ref()] });
  await h.prompt('second', 'group-a', { references: [ref(), ref(2, 'new source details')] });
  assert.equal((await h.state()).references.length, 2); prefixes(h.wires);
  const old = await h.state();
  await assert.rejects(h.handle('compact', { sessionId: 'group-a', expectedRevision: old.revision - 1, summary: 'summary' }), { code: 'REVISION_CONFLICT' });
  const compacted = await h.handle('compact', { sessionId: 'group-a', expectedRevision: old.revision, summary: 'Business facts and unresolved question.' });
  assert.deepEqual(compacted.references, []);
  assert.equal(compacted.messages.some((m) => m.customType === 'maibot.reference'), false);
  const duplicate = await h.handle('append', { sessionId: 'group-a', additions: [user('first')] });
  assert.deepEqual(duplicate.acceptedSourceIds, ['first']); assert.equal(strings(duplicate).includes('first'), false);
  await h.prompt('third', 'group-a', { references: [ref()] });
  assert.equal((await h.state()).references.length, 1);
  assert.equal(JSON.stringify(h.wires.at(-1)).match(/source details/g)?.length, 1);
});

test('failed, canceled, and concurrent compactions retain original prefix and newly admitted messages', async (t) => {
  let mode = 'failure'; const entered = deferred(), release = deferred();
  const h = harness(t, { summarize: async ({ signal }) => {
    if (mode === 'failure') throw new Error('summary provider failed');
    entered.resolve();
    if (mode === 'cancel') await new Promise((_yes, no) => signal.addEventListener('abort', () => no(new Error('canceled')), { once: true }));
    else await release.promise;
    return 'business summary';
  } });
  await h.initialize(); await h.prompt('first'); await h.prompt('second'); const old = await h.state();
  await assert.rejects(h.handle('compact', { sessionId: 'group-a' }), /summary provider failed/);
  assert.deepEqual((await h.state()).messages, old.messages);
  mode = 'cancel'; const cancelled = h.handle('compact', { sessionId: 'group-a' }); await entered.promise;
  const rejected = assert.rejects(cancelled, /canceled|cancelled/i);
  await h.handle('abort', { sessionId: 'group-a' }); await rejected;
  assert.deepEqual((await h.state()).messages, old.messages);
  mode = 'concurrent'; const compact = h.handle('compact', { sessionId: 'group-a' });
  // compact() reaches its awaited hook on the next microtask.
  await new Promise((done) => setImmediate(done));
  const incoming = await h.handle('append', { sessionId: 'group-a', additions: [user('during summary')] });
  assert.deepEqual(incoming.acceptedSourceIds, ['during summary']);
  release.resolve(); await assert.rejects(compact, /New input arrived/);
  const current = await h.state(); assert.deepEqual(current.messages.slice(0, old.messages.length), old.messages);
  assert.ok(strings(current).includes('during summary'));
});

for (const failure of ['user', 'assistant', 'toolResult', 'compaction']) {
  test(`append/fsync failure at ${failure} poisons session before any later wire or side effect`, async (t) => {
    let fail = false;
    const h = harness(t, { response: (n) => failure === 'toolResult' && n === 1 ? [call('send', {}, 'a', 'fa'), call('send', {}, 'b', 'fb')]
      : failure === 'assistant' && n === 1 ? [call('send', {}, 'a', 'fa')] : undefined,
    beforePersist: (entry) => { if (fail && (entry.type === failure || entry.message?.role === failure)) throw new Error('simulated disk full'); } });
    await h.initialize('group-a', [tool('send')]);
    if (failure === 'compaction') { await h.prompt('one'); await h.prompt('two'); }
    fail = true;
    await assert.rejects(failure === 'compaction' ? h.handle('compact', { sessionId: 'group-a', summary: 'summary' }) : h.prompt(), /disk full/);
    const before = h.wires.length;
    await assert.rejects(h.prompt('must not continue'), { code: 'PERSISTENCE_FAILED' });
    assert.equal(h.wires.length, before); assert.match((await h.state()).poisoned, /disk full/);
    assert.equal(h.calls.length, failure === 'toolResult' ? 1 : 0);
  });
}

test('overflow errors and transient provider failures remain archived without automatic retry or context_edit', async (t) => {
  const h = harness(t, { response: (n) => n === 1
    ? new Response(JSON.stringify({ error: { message: 'maximum context length exceeded', code: 'context_length_exceeded' } }), { status: 400, headers: { 'content-type': 'application/json' } }) : undefined });
  await h.initialize(); const failed = await h.prompt('original');
  assert.equal(h.wires.length, 1); assert.ok(failed.error);
  assert.equal(h.runtime.sessions.get('group-a').manager.getEntries().filter((entry) => ['context_edit', 'compaction'].includes(entry.type)).length, 0);
  await h.prompt('retry after explicit user input');
  assert.deepEqual((await h.state()).messages.slice(0, failed.messages.length), failed.messages);
  assert.equal(h.wires.length, 2); assert.equal((await h.state()).poisoned, null);
});

test('public archive reader redacts private reasoning, signatures, system and internal metadata', async (t) => {
  const h = harness(t, { response: () => [reasoning(), answer('public conclusion')] });
  await h.initialize(); await h.prompt('public input', 'group-a', { references: [ref()] });
  const state = await h.state();
  const read = await h.handle('read_archive', { sessionId: 'group-a', eventIds: state.archive.eventIds, limit: 12000 });
  assert.ok(read.text.includes('public conclusion')); assert.ok(read.text.includes('source details'));
  assert.equal(read.text.includes('opaque-encrypted'), false); assert.equal(read.text.includes('fixture summary'), false);
  assert.equal(read.text.includes('Fixed group persona'), false); assert.equal(read.text.includes('fixture-only'), false);
});

test('frozen system/tools/route reject restart drift; torn JSONL fails instead of silently losing accepted records', async (t) => {
  const h = harness(t); await h.initialize(); await h.prompt(); const file = h.runtime.sessions.get('group-a').manager.getSessionFile();
  await h.handle('close', { sessionId: 'group-a' });
  await assert.rejects(h.initialize('group-a', [], { system: 'new persona' }), { code: 'SESSION_CONTRACT_CHANGED' });
  await assert.rejects(h.initialize('group-a', [tool('new')]), { code: 'SESSION_CONTRACT_CHANGED' });
  appendFileSync(file, '{"incomplete"');
  await assert.rejects(h.initialize(), { code: 'CORRUPT_SESSION' });
});

test('single-writer lock excludes another runtime and orderly concurrent close releases it exactly once', async (t) => {
  const a = harness(t); await a.initialize();
  const b = harness(t, { root: a.dir });
  await assert.rejects(b.initialize(), { code: 'SESSION_LOCKED' });
  const closed = await Promise.all([a.handle('close', { sessionId: 'group-a' }), a.handle('close', { sessionId: 'group-a' })]);
  assert.ok(closed.every((item) => item.closed));
  await b.initialize();
  assert.equal((await a.handle('close', { sessionId: 'group-a' })).alreadyClosed, true);
});


test('default business summary uses a real isolated SDK session and never coding summary instructions', async (t) => {
  const h = harness(t); await h.initialize();
  await h.prompt('first'); await h.prompt('second');
  const state = await h.handle('compact', { sessionId: 'group-a' });
  assert.equal(h.wires.length, 3);
  assert.equal(state.messages.filter((message) => message.role === 'compactionSummary').length, 1);
  assert.equal(h.wires[2].tools?.length ?? 0, 0);
  assert.ok(JSON.stringify(h.wires[2]).includes('Summarize this group conversation faithfully'));
  assert.equal(JSON.stringify(h.wires[2]).includes('## Goal'), false);
  assert.equal(JSON.stringify(h.wires[2]).includes('## Next Steps'), false);
});

test('fsync failure after bytes were appended never ACKs; restart deduplicates durable source rather than replaying it', async (t) => {
  let fail = false;
  const h = harness(t, { syncFile: (fd) => { if (fail) throw new Error('simulated fsync failure'); fsyncSync(fd); } });
  await h.initialize(); fail = true;
  await assert.rejects(h.handle('append', { sessionId: 'group-a', additions: [user('uncertain sync', 'durable-source')] }), { code: 'PERSISTENCE_FAILED' });
  await h.handle('close', { sessionId: 'group-a' }); fail = false;
  await h.initialize();
  const accepted = await h.handle('append', { sessionId: 'group-a', additions: [user('uncertain sync', 'durable-source')] });
  assert.deepEqual(accepted.acceptedSourceIds, ['durable-source']);
  assert.equal(strings(accepted).filter((text) => text === 'uncertain sync').length, 1);
});

test('real orphan call accepts only matching known receipt; pending references wait until the result is durably paired', async (t) => {
  const h = harness(t); await h.initialize('group-a', [tool('send')]);
  const owned = h.runtime.sessions.get('group-a');
  owned.manager.appendMessage({ role: 'assistant', content: [{ type: 'toolCall', id: 'orphan', name: 'send', arguments: {} }],
    api: owned.resolved.model.api, provider: owned.resolved.model.provider, model: owned.resolved.model.id,
    stopReason: 'toolUse', timestamp: 2, usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 } });
  await h.handle('close', { sessionId: 'group-a' }); await h.initialize('group-a', [tool('send')]);
  await assert.rejects(h.prompt('queued', 'group-a', { references: [ref()] }), { code: 'UNRESOLVED_TOOL_CALLS' });
  assert.equal((await h.state()).references.length, 0);
  await assert.rejects(h.handle('resolve_tools', { sessionId: 'group-a', results: [{ role: 'toolResult', toolCallId: 'orphan', toolName: 'wrong', content: [], isError: false }] }), { code: 'INVALID_TOOL_RESULT' });
  const resumed = await h.handle('resolve_tools', { sessionId: 'group-a', results: [{ role: 'toolResult', toolCallId: 'orphan', toolName: 'send', content: [{ type: 'text', text: 'known receipt' }], isError: false, timestamp: 3 }] });
  const index = resumed.messages.findIndex((message) => message.role === 'assistant');
  assert.equal(resumed.messages[index + 1].toolCallId, 'orphan');
  assert.equal(resumed.references.length, 1);
  assert.equal(h.calls.length, 0); assert.equal(h.wires.length, 0);
});

test('unexpected writer-lock subprocess loss poisons live session before another write or wire', async (t) => {
  const h = harness(t); await h.initialize();
  const owned = h.runtime.sessions.get('group-a');
  process.kill(owned.releaseLock.pid, 'SIGKILL');
  await new Promise((done) => setTimeout(done, 30));
  assert.match((await h.state()).poisoned, /writer lock/);
  await assert.rejects(h.prompt(), { code: 'PERSISTENCE_FAILED' });
  assert.equal(h.wires.length, 0);
});


test('RPC snapshots and event notifications never export native reasoning or a lifetime transcript', async (t) => {
  const h = harness(t, { response: () => [reasoning(), answer('public answer')] }); await h.initialize();
  const result = await h.runtime.handle('prompt', { sessionId: 'group-a', additions: [user('hello')] });
  assert.equal(result.messages, undefined); assert.equal(result.conclusion, 'public answer');
  assert.equal(JSON.stringify(result).includes('opaque-encrypted'), false);
  assert.equal(JSON.stringify(h.events).includes('opaque-encrypted'), false);
  assert.equal(JSON.stringify(h.events).includes('fixture summary'), false);
  assert.equal(JSON.stringify(h.events).includes('thinking'), false);
  assert.ok(h.events.some(({ event }) => event.type === 'message_end' && event.message.content.some((part) => part.text === 'public answer')));
  assert.ok(result.archive.recentEventIds.length <= 20);
});

test('lost tool callback ACK recovery retains image receipt content without duplicate function outputs', async (t) => {
  const image = { type: 'image', data: 'aGVsbG8=', mimeType: 'image/png' };
  const config = fixtureConfig(); config.models[0].visual = true;
  const h = harness(t, { response: (n) => n === 1 ? [call('fetch', {}, 'image-call', 'fi')] : undefined,
    execute: async () => { throw new Error('receipt was durable but callback ACK was lost'); } });
  await h.initialize('group-a', [tool('fetch')], { config }); await h.prompt();
  const recovered = await h.handle('resolve_tools', { sessionId: 'group-a', results: [{ role: 'toolResult', toolName: 'fetch',
    toolCallId: 'image-call|fi', content: [{ type: 'text', text: 'source image' }, image], isError: false, timestamp: 2 }] });
  assert.ok(recovered.messages.find((message) => message.customType === 'maibot.receipt_reconciled').content.some((part) => part.data === image.data));
  assert.equal(recovered.messages.filter((message) => message.role === 'toolResult').length, 1);
  await h.prompt('read confirmed image');
  assert.ok(JSON.stringify(h.wires[1]).includes('data:image/png;base64,aGVsbG8='));
  assert.equal(h.calls.length, 1);
});

test('successful compaction exposes a bounded scoped archive handle with restart-safe public pagination', async (t) => {
  const h = harness(t); await h.initialize(); await h.prompt('original source evidence'); await h.prompt('followup');
  const state = await h.handle('compact', { sessionId: 'group-a', summary: 'Business summary without manually copied event identifiers.' });
  const summary = state.messages.find((message) => message.role === 'compactionSummary').summary;
  const handle = summary.match(/pi-archive:[0-9a-f]+/)[0];
  const first = await h.handle('read_archive', { sessionId: 'group-a', eventIds: [handle], offset: 0, limit: 12000 });
  assert.ok(first.text.includes('original source evidence'));
  const page = await h.handle('read_archive', { sessionId: 'group-a', eventIds: [handle], offset: 0, limit: 30 });
  assert.equal(page.next_offset, 30);
  await h.handle('close', { sessionId: 'group-a' }); await h.initialize();
  assert.deepEqual(await h.handle('read_archive', { sessionId: 'group-a', eventIds: [handle], offset: 0, limit: 30 }), page);
  await h.initialize('other');
  await assert.rejects(h.handle('read_archive', { sessionId: 'other', eventIds: [handle] }), { code: 'ARCHIVE_NOT_FOUND' });
});

test('compaction rejects a cut inside a tool pair without wedging the transition', async (t) => {
  const h = harness(t, { response: (n) => n === 1 ? [call('fetch', {})] : undefined });
  await h.initialize('group-a', [tool('fetch')]); await h.prompt('first'); await h.prompt('second');
  const owned = h.runtime.sessions.get('group-a');
  const split = owned.manager.getEntries().find((entry) => entry.type === 'message' && entry.message.role === 'toolResult').id;
  await assert.rejects(h.handle('compact', { sessionId: 'group-a', summary: 'summary', firstKeptEntryId: split }), { code: 'INVALID_COMPACTION' });
  assert.equal((await h.state()).isCompacting, false);
  await assert.rejects(h.handle('compact', { sessionId: 'group-a', summary: 'summary', firstKeptEntryId: 'not-owned' }), { code: 'INVALID_COMPACTION' });
  await h.prompt('not wedged');
});


test('pressure metrics are Node-owned advisory values and provider failure diagnostics stay private', async (t) => {
  const h = harness(t, { response: (n) => n === 2
    ? new Response(JSON.stringify({ error: { message: 'private provider detail secret-123', code: 'bad_request' } }), { status: 400, headers: { 'content-type': 'application/json' } }) : undefined });
  await h.initialize();
  const success = await h.runtime.handle('prompt', { sessionId: 'group-a', additions: [user('hello')] });
  assert.ok(success.visibleChars > 0);
  assert.equal(success.latestReportedInputTokens, 80);
  assert.equal(success.pressureIsHardBound, false);
  const failure = await h.runtime.handle('prompt', { sessionId: 'group-a', additions: [user('second')] });
  assert.equal(failure.error, 'Pi provider request did not complete successfully');
  assert.equal(JSON.stringify(failure).includes('secret-123'), false);
  assert.equal(JSON.stringify(h.events).includes('secret-123'), false);
});


for (const timeoutSource of ['provider', 'task']) {
  test(`${timeoutSource} response timeout surfaces safe incomplete state without retry and retains the original source`, async (t) => {
    const config = fixtureConfig();
    if (timeoutSource === 'provider') config.api_providers[0].timeout = 0.03;
    else config.model_task_config.planner.hard_timeout = 0.03;
    const h = harness(t, { response: async (_number, _wire, signal) => {
      await new Promise((_yes, no) => {
        if (signal.aborted) no(new DOMException('aborted', 'AbortError'));
        else signal.addEventListener('abort', () => no(new DOMException('aborted', 'AbortError')), { once: true });
      });
    } });
    await h.initialize('group-a', [], { config });
    const result = await h.prompt('must remain durable');
    assert.equal(h.wires.length, 1);
    assert.equal(result.stopReason, 'aborted');
    assert.equal(result.error, 'Pi provider request was canceled or timed out');
    assert.deepEqual(strings(result), ['must remain durable']);
    assert.equal(h.runtime.sessions.get('group-a').manager.getEntries().filter((entry) => ['compaction', 'context_edit'].includes(entry.type)).length, 0);
  });
}

test('truncated main response does not execute partial tool proposals or automatically retry the same output budget', async (t) => {
  const h = harness(t, { response: async () => {
    const response = fixtureResponse([call('send', {}, 'truncated', 'ft')]);
    const body = (await response.text()).replace('"type":"response.completed"', '"type":"response.incomplete"')
      .replace('"status":"completed","output":', '"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":');
    return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
  } });
  await h.initialize('group-a', [tool('send')]);
  const result = await h.prompt();
  assert.equal(result.stopReason, 'length');
  assert.match(result.error, /output limit/);
  assert.equal(h.wires.length, 1); assert.equal(h.calls.length, 0);
  assert.ok(result.messages.some((message) => message.role === 'toolResult' && message.isError));
});

test('maintenance never summarizes the first unprocessed batch or a failed first turn', async (t) => {
  const h = harness(t, { response: () => new Response('{"error":{"message":"failed"}}', { status: 400 }) });
  await h.initialize();
  const source = user('first request must stay verbatim');
  await h.handle('append', { sessionId: 'group-a', additions: [source] });
  const before = await h.state();
  const skipped = await h.handle('compact', { sessionId: 'group-a', preserveUnprocessed: true });
  assert.equal(skipped.compacted, false); assert.equal(skipped.reason, 'no_completed_turn');
  assert.deepEqual(skipped.messages, before.messages); assert.equal(h.wires.length, 0);
  await h.handle('prompt', { sessionId: 'group-a' });
  assert.equal((await h.handle('compact', { sessionId: 'group-a', preserveUnprocessed: true })).compacted, false);
  assert.equal(h.wires.length, 1);
});

test('maintenance uses a restart-safe completed boundary and preserves new text, image and reference tail exactly', async (t) => {
  const config = fixtureConfig(); config.models[0].visual = true;
  const h = harness(t, { response: (number) => number === 2 ? [answer('Summary of only the already completed conversation.')] : undefined });
  await h.initialize('group-a', [], { config }); await h.prompt('completed old source');
  const source = { ...user('unused', 'fresh-image'), content: [
    { type: 'text', text: 'brand-new image question must not be summarized' },
    { type: 'image', data: 'aGVsbG8=', mimeType: 'image/png' },
  ] };
  await h.handle('append', { sessionId: 'group-a', additions: [source, user('brand-new plain text', 'fresh-text')] });
  h.runtime.sessions.get('group-a').appendReferences([ref(8, 'brand-new retrieved evidence')]);
  const tail = (await h.state()).messages.slice(-3);
  await h.handle('close', { sessionId: 'group-a' }); await h.initialize('group-a', [], { config });
  const compacted = await h.handle('compact', { sessionId: 'group-a', preserveUnprocessed: true });
  assert.equal(compacted.compacted, true);
  assert.deepEqual(compacted.messages.slice(-3), tail);
  const nonSystem = compacted.messages.filter((message) => message.role !== 'system');
  assert.equal(nonSystem[0].role, 'compactionSummary');
  assert.equal(nonSystem.length, 4);
  assert.ok(JSON.stringify(h.wires[1]).includes('completed old source'));
  assert.equal(JSON.stringify(h.wires[1]).includes('brand-new'), false);
  assert.equal(JSON.stringify(h.wires[1]).includes('aGVsbG8='), false);
  const skipped = await h.handle('compact', { sessionId: 'group-a', preserveUnprocessed: true });
  assert.equal(skipped.compacted, false); assert.equal(skipped.reason, 'no_processed_prefix');
  await h.handle('prompt', { sessionId: 'group-a' });
  const nextWire = JSON.stringify(h.wires[2]);
  assert.ok(nextWire.includes('data:image/png;base64,aGVsbG8='));
  for (const text of ['brand-new image question', 'brand-new plain text', 'brand-new retrieved evidence']) {
    assert.equal(nextWire.split(text).length - 1, 1);
  }
});

test('maintenance preserves arrivals queued during a terminating tool after the last completed batch', async (t) => {
  const h = harness(t, { response: () => [call('wait', {}, 'wait-call', 'fw')],
    execute: async (_params, _signal, runtime) => {
      await runtime.handle('append', { sessionId: 'group-a', additions: [user('arrived during wait')] });
      return success('wait accepted', { terminate: true });
    } });
  await h.initialize('group-a', [tool('wait')]); await h.prompt('processed request');
  const compacted = await h.handle('compact', { sessionId: 'group-a', preserveUnprocessed: true, summary: 'Previously completed wait.' });
  assert.equal(compacted.compacted, true);
  assert.deepEqual(strings(compacted), ['arrived during wait']);
  assert.equal(compacted.messages.some((message) => message.role === 'toolResult'), false);
});
