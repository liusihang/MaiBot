import assert from 'node:assert/strict';
import test from 'node:test';
import { Agent } from '@earendil-works/pi-agent-core';
import { PiRuntime } from '../runtime.mjs';

const user = (text, id = text) => ({ role: 'user', content: text, timestamp: 1, hostEventId: id });
const tool = (name) => ({ name, description: `Host ${name}`, parameters: { type: 'object', properties: {}, additionalProperties: true } });
const call = (id, name, args = {}) => ({ type: 'toolCall', id, name, arguments: args });
const reply = (content, extra = {}) => ({ role: 'assistant', content, timestamp: 2, ...extra });
const textReply = (text, extra = {}) => reply([{ type: 'text', text }], extra);
const success = (text = 'ok', extra = {}) => ({ content: [{ type: 'text', text }], details: {}, ...extra });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const clone = (value) => structuredClone(value);
const init = (runtime, sessionId = 'group-a', tools = [], extra = {}) => runtime.handle('init', { sessionId, system: 'Fixed MaiBot persona', tools, ...extra });
const prompt = (runtime, text = 'hello', sessionId = 'group-a') => runtime.handle('prompt', { sessionId, additions: [user(text)] });
const state = (runtime, sessionId = 'group-a') => runtime.handle('state', { sessionId });

function harness({ model, execute = async () => success(), event = async () => {} }) {
  const events = [], models = [], tools = [];
  const runtime = new PiRuntime(async (method, params, signal) => {
    if (method === 'event') {
      events.push(clone(params));
      return event(params, runtime);
    }
    if (method === 'model') {
      models.push(clone(params));
      return model(params, signal, runtime);
    }
    if (method === 'tool') {
      tools.push(clone(params));
      return execute(params, signal, runtime);
    }
    assert.fail(`Unexpected callback ${method}`);
  });
  return { runtime, events, models, tools };
}

function assertPrefixes(contexts) {
  for (let i = 1; i < contexts.length; i++) {
    assert.deepEqual(contexts[i].messages.slice(0, contexts[i - 1].messages.length), contexts[i - 1].messages);
  }
}

test('actual installed Pi Agent runs search → read → reply, preserving native and multi-turn prefixes', async () => {
  const native = {
    items: [{ _type: 'ChatMessage', content: [
      { _type: 'SignedReasoning', text: 'not reconstructed', signature: 'sig:+/==', unknown: { exact: [false, null, 0] } },
      { _type: 'FunctionCall', id: 'search-1', arguments: '{ "q": "x", "n": 1 }' },
    ] }], envelope: ['future-provider-field', { keep: true }],
  };
  const plan = [
    reply([call('search-1', 'search')], { native }),
    reply([call('read-1', 'read', { ref: 'source' })], { native: { items: [{ raw: 'read plan' }] } }),
    reply([call('reply-1', 'reply', { text: 'answer' })], { native: { items: [{ raw: 'reply plan' }] } }),
    textReply('second turn', { native: { items: [{ raw: 'second response' }] } }),
  ];
  const h = harness({ model: () => plan.shift(), execute: ({ name, toolCallId }) => success(name, {
    details: { native: { items: [{ _type: 'FunctionCallOutput', id: toolCallId, output: name }] } },
    terminate: name === 'reply',
  }) });
  const initial = await init(h.runtime, 'group-a', ['search', 'read', 'reply'].map(tool));
  assert.ok(h.runtime.sessions.get('group-a').agent instanceof Agent);
  const originalAgent = h.runtime.sessions.get('group-a').agent;
  assert.equal(initial.messages[0].role, 'system');
  const first = await prompt(h.runtime);
  assert.deepEqual(h.tools.map((item) => item.name), ['search', 'read', 'reply']);
  assert.equal(h.models.length, 3);
  assert.deepEqual(first.messages.find((m) => m.role === 'assistant').native, native);
  assert.deepEqual(h.models[1].messages.find((m) => m.role === 'assistant').native, native);
  assert.deepEqual(h.models[2].messages.find((m) => m.toolCallId === 'search-1').details.native,
    { items: [{ _type: 'FunctionCallOutput', id: 'search-1', output: 'search' }] });
  const second = await prompt(h.runtime, 'another group message');
  assert.strictEqual(h.runtime.sessions.get('group-a').agent, originalAgent);
  assert.deepEqual(second.messages.slice(0, first.messages.length), first.messages);
  assertPrefixes(h.models);
  assert.ok(h.models.every((request) => request.system === 'Fixed MaiBot persona'));
  assert.deepEqual(h.events.map((entry) => entry.revision), h.events.map((_entry, index) => index + 1));
});

test('groups run independently without leaking context or tools', async () => {
  const blockers = new Map([['group-a', deferred()], ['group-b', deferred()]]);
  const started = new Map([['group-a', deferred()], ['group-b', deferred()]]);
  const h = harness({ model: async ({ sessionId }) => {
    started.get(sessionId).resolve();
    await blockers.get(sessionId).promise;
    return textReply(sessionId);
  } });
  await init(h.runtime, 'group-a', [tool('only_a')]);
  await init(h.runtime, 'group-b', [tool('only_b')]);
  const a = prompt(h.runtime, 'a secret', 'group-a');
  const b = prompt(h.runtime, 'b secret', 'group-b');
  await Promise.all([...started.values()].map((entry) => entry.promise));
  blockers.get('group-b').resolve();
  const doneB = await b;
  assert.equal((await state(h.runtime, 'group-a')).isRunning, true);
  blockers.get('group-a').resolve();
  const doneA = await a;
  assert.equal(JSON.stringify(doneA.messages).includes('b secret'), false);
  assert.equal(JSON.stringify(doneB.messages).includes('a secret'), false);
  assert.deepEqual(h.models.find((p) => p.sessionId === 'group-a').tools.map((t) => t.name), ['only_a']);
  assert.deepEqual(h.models.find((p) => p.sessionId === 'group-b').tools.map((t) => t.name), ['only_b']);
});

test('assistant and tool-result persistence are barriers before sequential side effects', async () => {
  const assistantSeen = deferred(), allowAssistant = deferred(), toolSeen = deferred(), allowTool = deferred();
  const order = [];
  let models = 0;
  const h = harness({
    model: () => ++models === 1 ? reply([call('a', 'first'), call('b', 'second')]) : textReply('done'),
    execute: ({ name }) => { order.push(name); return success(); },
    event: async ({ event }) => {
      if (event.type === 'message_end' && event.message.role === 'assistant' && event.message.content[0]?.type === 'toolCall') {
        assistantSeen.resolve(); await allowAssistant.promise;
      }
      if (event.type === 'message_end' && event.message.toolCallId === 'a') {
        toolSeen.resolve(); await allowTool.promise;
      }
    },
  });
  await init(h.runtime, 'group-a', [tool('first'), tool('second')]);
  const running = prompt(h.runtime);
  await assistantSeen.promise;
  assert.deepEqual(order, []);
  allowAssistant.resolve();
  await toolSeen.promise;
  assert.deepEqual(order, ['first']);
  allowTool.resolve();
  await running;
  assert.deepEqual(order, ['first', 'second']);
});

test('group messages arriving mid-tool or reentrantly in message_end are queued without deadlock or loss', async () => {
  const executing = deferred(), finish = deferred();
  let count = 0, ingested = false;
  const h = harness({
    model: () => ++count === 1 ? reply([call('read-a', 'read')]) : textReply('done'),
    execute: async () => { executing.resolve(); await finish.promise; return success('document'); },
    event: async ({ event }, runtime) => {
      if (!ingested && event.type === 'message_end' && event.message.role === 'toolResult') {
        ingested = true;
        await runtime.handle('append', { sessionId: 'group-a', additions: [user('reentrant arrival')] });
      }
    },
  });
  await init(h.runtime, 'group-a', [tool('read')]);
  const running = prompt(h.runtime);
  await executing.promise;
  const queued = await prompt(h.runtime, 'mid-tool arrival');
  assert.equal(queued.queued, true);
  finish.resolve();
  const end = await running;
  assert.equal(end.queuedMessages.length, 0);
  assert.deepEqual(end.messages.filter((m) => m.role === 'user').map((m) => m.content),
    ['hello', 'mid-tool arrival', 'reentrant arrival']);
  assert.deepEqual(h.models[1].messages.filter((m) => m.role === 'user').map((m) => m.content),
    ['hello', 'mid-tool arrival', 'reentrant arrival']);
  assertPrefixes(h.models);
});

test('wait/stop terminate the run without an extra provider call and retain queued input', async () => {
  let count = 0;
  const h = harness({
    model: () => ++count === 1 ? reply([call('wait-a', 'wait'), call('danger', 'send')]) : textReply('awake'),
    execute: async ({ name }, _signal, runtime) => {
      assert.equal(name, 'wait');
      await runtime.handle('append', { sessionId: 'group-a', additions: [user('pending while waiting')] });
      return success('wait registered', { terminate: true });
    },
  });
  await init(h.runtime, 'group-a', [tool('wait'), tool('send')]);
  const end = await prompt(h.runtime);
  assert.equal(h.models.length, 1);
  assert.equal(h.tools.length, 1);
  assert.equal(end.queuedMessages.length, 1);
  assert.equal(end.messages.find((m) => m.toolCallId === 'danger').isError, true);
  assert.equal(end.paused, false);
  const resumed = await h.runtime.handle('continue', { sessionId: 'group-a' });
  assert.equal(h.models.length, 2);
  assert.equal(resumed.queuedMessages.length, 0);
  assert.equal(resumed.messages.at(-1).content[0].text, 'awake');
});

test('tool exceptions and provider exceptions are durable Pi errors, not swallowed success', async () => {
  let count = 0;
  const h = harness({
    model: () => {
      if (++count === 1) return reply([call('bad-a', 'broken')]);
      throw new Error('provider offline');
    },
    execute: async () => { throw new Error('read failed'); },
  });
  await init(h.runtime, 'group-a', [tool('broken')]);
  const end = await prompt(h.runtime);
  assert.equal(end.messages.find((m) => m.toolCallId === 'bad-a').isError, true);
  assert.equal(end.messages.at(-1).stopReason, 'error');
  assert.equal(end.error, 'provider offline');
  assert.ok(h.events.some(({ event }) => event.type === 'message_end' && event.message.errorMessage === 'provider offline'));
});

test('abort records unfinished sequential calls and fails closed until explicit host recovery', async () => {
  const executing = deferred();
  let count = 0;
  const h = harness({
    model: () => ++count === 1 ? reply([call('one', 'send'), call('two', 'send')]) : textReply('recovered'),
    execute: async (_params, signal) => {
      executing.resolve();
      await new Promise((_resolve, reject) => signal.addEventListener('abort',
        () => reject(new DOMException('Operation aborted', 'AbortError')), { once: true }));
      assert.fail('aborted execution must not finish normally');
    },
  });
  await init(h.runtime, 'group-a', [tool('send')]);
  const running = prompt(h.runtime);
  await executing.promise;
  const cancelled = await h.runtime.handle('abort', { sessionId: 'group-a' });
  await running;
  assert.equal(h.models.length, 1);
  assert.equal(h.tools.length, 1);
  assert.equal(cancelled.paused, true);
  assert.deepEqual(cancelled.unresolvedToolCalls.map((c) => c.id), ['two']);
  await assert.rejects(prompt(h.runtime, 'cannot silently resume'), { code: 'UNRESOLVED_TOOL_CALLS' });
  await h.runtime.handle('append', { sessionId: 'group-a', additions: [user('durable later input')] });
  const before = await state(h.runtime);
  const recovered = await h.runtime.handle('resolve_tools', { sessionId: 'group-a', expectedRevision: before.revision,
    results: [{ role: 'toolResult', toolCallId: 'two', toolName: 'send', content: [{ type: 'text', text: 'not_started: canceled before dispatch' }], isError: true }],
  });
  assert.equal(recovered.paused, false);
  await h.runtime.handle('continue', { sessionId: 'group-a' });
  assert.equal(h.models.length, 2);
  assert.ok(h.events.some(({ event }) => event.type === 'run_aborted'));
});

test('pre-aborted stream never invokes the host provider, and model abort is persisted', async () => {
  const begun = deferred();
  const h = harness({ model: async (_params, signal) => {
    begun.resolve();
    await new Promise((_resolve, reject) => signal.addEventListener('abort',
      () => reject(new DOMException('Operation aborted', 'AbortError')), { once: true }));
  } });
  await init(h.runtime);
  const running = prompt(h.runtime);
  await begun.promise;
  const end = await h.runtime.handle('abort', { sessionId: 'group-a' });
  await running;
  assert.equal(end.messages.at(-1).stopReason, 'aborted');
  const alreadyAborted = new AbortController();
  alreadyAborted.abort();
  const response = h.runtime.sessions.get('group-a').modelStream({ api: 'x', provider: 'x', id: 'x' }, { messages: [] }, { signal: alreadyAborted.signal });
  assert.equal((await response.result()).stopReason, 'aborted');
  assert.equal(h.models.length, 1);
});

test('dynamic tools append declarations only, preserve old history, and work on the next Pi turn', async () => {
  let count = 0;
  const h = harness({
    model: () => {
      count++;
      if (count === 1) return reply([call('discover-a', 'discover')]);
      if (count === 2) return reply([call('new-a', 'read_new')]);
      return textReply('done');
    },
    execute: async ({ name }, _signal, runtime) => {
      if (name === 'discover') await runtime.handle('add_tools', { sessionId: 'group-a', tools: [tool('read_new')] });
      return success();
    },
  });
  const initial = await init(h.runtime, 'group-a', [tool('discover')]);
  const end = await prompt(h.runtime);
  assert.deepEqual(h.tools.map((item) => item.name), ['discover', 'read_new']);
  assert.deepEqual(end.messages[0], initial.messages[0]);
  assert.deepEqual(h.models[1].tools.map((t) => t.name), ['discover', 'read_new']);
  const delta = end.messages.find((m, i) => i > 0 && m.role === 'system');
  assert.equal(delta.content, '');
  assert.deepEqual(delta.toolsAdded.map((t) => t.name), ['read_new']);
  assertPrefixes(h.models);
  await assert.rejects(h.runtime.handle('add_tools', { sessionId: 'group-a', tools: [{ ...tool('discover'), description: 'changed' }] }), { code: 'TOOL_IMMUTABLE' });
  await assert.rejects(h.runtime.handle('prompt', { sessionId: 'group-a', additions: [{ role: 'system', content: 'changed' }] }), { code: 'INVALID_MESSAGES' });
});

test('restart hydration preserves native messages exactly and does not reconstruct earlier outputs', async () => {
  const native = { items: [{ opaque: { signed: '+/==', tool_call_json: '{  "a" : 2 }' } }] };
  const h = harness({ model: () => textReply('first', { native }) });
  await init(h.runtime);
  const original = await prompt(h.runtime);
  const h2 = harness({ model: () => textReply('second') });
  const replay = await init(h2.runtime, 'group-a', original.tools, { messages: original.messages, revision: original.revision });
  assert.deepEqual(replay.messages, original.messages);
  await prompt(h2.runtime, 'new event');
  assert.deepEqual(h2.models[0].messages.slice(0, original.messages.length), original.messages);
});

test('compaction is explicit CAS, reserves the transition, and cannot split host and Pi views', async () => {
  const entered = deferred(), allow = deferred();
  const h = harness({ model: () => textReply('done'), event: async ({ event }) => {
    if (event.type === 'context_replaced') { entered.resolve(); await allow.promise; }
  } });
  await init(h.runtime);
  const before = await prompt(h.runtime);
  await assert.rejects(h.runtime.handle('replace_context', { sessionId: 'group-a', expectedRevision: before.revision - 1, reason: 'summary', messages: before.messages }), { code: 'REVISION_CONFLICT' });
  const replacement = [before.messages[0], user('explicit summary')];
  const compacting = h.runtime.handle('replace_context', { sessionId: 'group-a', expectedRevision: before.revision, reason: 'host-requested compaction', messages: replacement });
  await entered.promise;
  await assert.rejects(prompt(h.runtime, 'racing arrival'), { code: 'SESSION_TRANSITION' });
  await assert.rejects(h.runtime.handle('append', { sessionId: 'group-a', additions: [user('racing queued arrival')] }), { code: 'SESSION_TRANSITION' });
  allow.resolve();
  const compacted = await compacting;
  assert.deepEqual(compacted.messages, replacement);
  await h.runtime.handle('append', { sessionId: 'group-a', additions: [user('retry preserved host event')] });
  await assert.rejects(h.runtime.handle('replace_context', { sessionId: 'group-a', expectedRevision: compacted.revision, reason: 'not allowed with pending input', messages: replacement }), { code: 'SESSION_BUSY' });
});

test('persistence failure poisons the session before side effects and never falls back to a local loop', async () => {
  const h = harness({ model: () => reply([call('send-a', 'send')]), event: async ({ event }) => {
    if (event.type === 'message_end' && event.message.role === 'assistant') throw new Error('disk full');
  } });
  await init(h.runtime, 'group-a', [tool('send')]);
  await assert.rejects(prompt(h.runtime), { code: 'PERSISTENCE_FAILED' });
  assert.equal(h.tools.length, 0);
  const end = await state(h.runtime);
  assert.match(end.poisoned, /disk full/);
  await assert.rejects(prompt(h.runtime, 'must not continue'), { code: 'PERSISTENCE_FAILED' });
});

test('maintenance request finishes every tool in its batch, then pauses until explicit compaction', async () => {
  let count = 0;
  const h = harness({
    model: () => ++count === 1
      ? reply([call('maintenance-one', 'read'), call('maintenance-two', 'read')], { hostMaintenanceRequired: true })
      : textReply('continued after compaction'),
  });
  await init(h.runtime, 'group-a', [tool('read')]);
  const end = await prompt(h.runtime);
  assert.equal(h.models.length, 1);
  assert.deepEqual(h.tools.map((item) => item.toolCallId), ['maintenance-one', 'maintenance-two']);
  assert.deepEqual(end.unresolvedToolCalls, []);
  assert.equal(end.maintenanceRequired, true);
  assert.equal(end.continuationRequired, true);
  await assert.rejects(h.runtime.handle('continue', { sessionId: 'group-a' }), { code: 'MAINTENANCE_REQUIRED' });
  await assert.rejects(prompt(h.runtime, 'must not bypass pressure stop'), { code: 'MAINTENANCE_REQUIRED' });
  const replacement = [end.messages[0], user('explicit host summary of completed batch')];
  const compacted = await h.runtime.handle('replace_context', { sessionId: 'group-a', expectedRevision: end.revision,
    reason: 'provider-reported context pressure', messages: replacement });
  assert.equal(compacted.maintenanceRequired, false);
  const resumed = await h.runtime.handle('continue', { sessionId: 'group-a' });
  assert.equal(resumed.messages.at(-1).content[0].text, 'continued after compaction');
  assert.equal(h.models.length, 2);
});

test('a new prompt after stop cannot jump ahead of earlier queued group messages', async () => {
  let count = 0;
  const h = harness({
    model: () => ++count === 1 ? reply([call('stop-one', 'stop')]) : textReply('received both in order'),
    execute: async (_params, _signal, runtime) => {
      await runtime.handle('append', { sessionId: 'group-a', additions: [user('older queued input')] });
      return success('stopped', { terminate: true });
    },
  });
  await init(h.runtime, 'group-a', [tool('stop')]);
  await prompt(h.runtime, 'first request');
  const end = await prompt(h.runtime, 'newer next request');
  assert.deepEqual(end.messages.filter((item) => item.role === 'user').map((item) => item.content),
    ['first request', 'older queued input', 'newer next request']);
  assert.equal(end.queuedMessages.length, 0);
});

test('maintenance does not ask host to continue a final answer or terminating wait tool', async () => {
  const h = harness({ model: () => textReply('final answer', { hostMaintenanceRequired: true }) });
  await init(h.runtime);
  const end = await prompt(h.runtime);
  assert.equal(end.maintenanceRequired, true);
  assert.equal(end.continuationRequired, false);
  const h2 = harness({
    model: () => reply([call('wait-last', 'wait')], { hostMaintenanceRequired: true }),
    execute: () => success('wait', { terminate: true }),
  });
  await init(h2.runtime, 'group-a', [tool('wait')]);
  const waited = await prompt(h2.runtime);
  assert.equal(waited.maintenanceRequired, true);
  assert.equal(waited.continuationRequired, false);
  assert.equal(waited.terminatedByTool, true);
});

test('concurrent shutdown paths close a session once and a late close is harmless', async () => {
  const entered = deferred(), release = deferred();
  const h = harness({ model: () => textReply('done'), event: async ({ event }) => {
    if (event.type === 'session_closed') { entered.resolve(); await release.promise; }
  } });
  await init(h.runtime);
  const first = h.runtime.handle('close', { sessionId: 'group-a' });
  await entered.promise;
  const second = h.runtime.handle('close', { sessionId: 'group-a' });
  release.resolve();
  assert.equal((await first).closed, true);
  assert.equal((await second).closed, true);
  assert.equal(h.events.filter(({ event }) => event.type === 'session_closed').length, 1);
  assert.equal((await h.runtime.handle('close', { sessionId: 'group-a' })).alreadyClosed, true);
});

test('a transient host provider callback failure remains auditable and does not poison the next input', async () => {
  let requests = 0;
  const h = harness({ model: () => {
    if (++requests === 1) throw new Error('temporary provider timeout');
    return textReply('recovered', { native: { items: [{ actual: 'provider output' }] } });
  } });
  await init(h.runtime);
  const failed = await prompt(h.runtime, 'original user');
  const receipt = failed.messages.at(-1);
  assert.equal(receipt.stopReason, 'error');
  assert.equal(receipt.hostSynthetic, 'provider_callback_failure');
  assert.equal(receipt.hostControl, 'provider_failure');
  assert.deepEqual(receipt.native, { items: [] });
  const resumed = await prompt(h.runtime, 'new user');
  assert.equal(resumed.error, null);
  assert.equal(resumed.messages.at(-1).content[0].text, 'recovered');
  assert.deepEqual(resumed.messages.slice(0, failed.messages.length), failed.messages);
  assert.deepEqual(h.models[1].messages.find((message) => message.hostSynthetic).native, { items: [] });
});

test('known host over-limit admission produces an explicit control receipt then resumes after compaction', async () => {
  let requests = 0;
  const h = harness({ model: () => ++requests === 1
    ? { maintenanceRequired: true, hostControl: 'capacity_admission', reason: 'verified request exceeds known capacity' }
    : textReply('resumed without inventing a provider response', { native: { items: [{ actual: 'provider output' }] } }),
  });
  await init(h.runtime);
  const end = await prompt(h.runtime, 'request that requires maintenance');
  assert.equal(end.maintenanceRequired, true);
  assert.equal(end.continuationRequired, true);
  assert.equal(end.error, null);
  const receipt = end.messages.at(-1);
  assert.equal(receipt.hostControl, 'capacity_admission');
  assert.equal(receipt.hostSynthetic, undefined);
  assert.deepEqual(receipt.content, []);
  assert.deepEqual(receipt.native, { items: [] });
  assert.equal(h.tools.length, 0);
  assert.deepEqual(end.messages.slice(0, 2), h.models[0].messages);
  // Preserve the explicit host receipt at the tail. Official Agent.continue()
  // rejects any assistant tail; the bridge may use Agent.prompt([]) at this
  // specific host-control boundary without adding or rewriting a user message.
  const replacement = [end.messages[0], user('explicit maintenance summary'), receipt];
  await h.runtime.handle('replace_context', { sessionId: 'group-a', expectedRevision: end.revision,
    reason: 'known admission over_limit', messages: replacement });
  const resumed = await h.runtime.handle('continue', { sessionId: 'group-a' });
  assert.equal(requests, 2);
  assert.deepEqual(resumed.messages.slice(0, replacement.length), replacement);
  assert.equal(resumed.messages.at(-1).content[0].text, 'resumed without inventing a provider response');
});
