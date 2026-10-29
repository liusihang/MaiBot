import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const sidecar = fileURLToPath(new URL('../sidecar.mjs', import.meta.url));
const user = (content) => ({ role: 'user', content, timestamp: 1 });
const tool = (name) => ({ name, description: name, parameters: { type: 'object' } });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

function client(t, callback) {
  const child = spawn(process.execPath, [sidecar], { stdio: ['pipe', 'pipe', 'pipe'] });
  const pending = new Map(), notifications = [], received = [], callbackErrors = [];
  let nextId = 0, stderr = '';
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  const command = (method, params) => {
    const id = `request-${++nextId}`;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      send({ id, method, params });
    });
  };
  const lineReader = createInterface({ input: child.stdout });
  lineReader.on('line', (line) => {
    const message = JSON.parse(line);
    received.push(message);
    if (message.method) {
      if (!message.id) { notifications.push(message); return; }
      void (async () => {
        try {
          const result = await callback(message, { command, send, child });
          send({ id: message.id, result });
        } catch (error) {
          callbackErrors.push(error);
          send({ id: message.id, error: { code: 'MOCK_ERROR', message: error.message } });
        }
      })();
      return;
    }
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id);
    if (message.error) item.reject(Object.assign(new Error(message.error.message), message.error));
    else item.resolve(message.result);
  });
  t.after(async () => {
    if (child.exitCode === null) {
      child.stdin.end();
      await once(child, 'exit');
    }
    assert.equal(stderr, '');
    assert.deepEqual(callbackErrors, []);
  });
  return { child, command, send, notifications, received };
}

test('JSONL subprocess routes concurrent groups and out-of-order callbacks while preserving opaque data', { timeout: 10000 }, async (t) => {
  const firstStarted = deferred(), secondStarted = deferred(), releaseFirst = deferred();
  const originalNative = { items: [{ _type: 'raw', content: { signature: '+/==', unknown: [null, false] } }] };
  const counts = new Map(), models = [];
  const c = client(t, async (message) => {
    if (message.method === 'event') return { committed: true };
    assert.equal(message.method, 'model');
    const { sessionId } = message.params;
    models.push(message.params);
    counts.set(sessionId, (counts.get(sessionId) ?? 0) + 1);
    if (sessionId === 'a' && counts.get(sessionId) === 1) {
      firstStarted.resolve();
      await releaseFirst.promise;
    } else if (sessionId === 'b') secondStarted.resolve();
    return { message: { role: 'assistant', content: [{ type: 'text', text: sessionId }], native: originalNative } };
  });
  await c.command('init', { sessionId: 'a', system: 'A', tools: [] });
  await c.command('init', { sessionId: 'b', system: 'B', tools: [] });
  const first = c.command('prompt', { sessionId: 'a', additions: [user('a input')] });
  await firstStarted.promise;
  const second = c.command('prompt', { sessionId: 'b', additions: [user('b input')] });
  await secondStarted.promise;
  const b = await second;
  assert.equal(b.messages.at(-1).content[0].text, 'b');
  releaseFirst.resolve();
  const a = await first;
  assert.deepEqual(a.messages.at(-1).native, originalNative);
  await c.command('prompt', { sessionId: 'a', additions: [user('another a')] });
  assert.deepEqual(models.at(-1).messages.find((item) => item.role === 'assistant').native, originalNative);
  assert.equal(models.filter((item) => item.sessionId === 'a').some((item) => JSON.stringify(item.messages).includes('b input')), false);
  const closed = await c.command('close', { sessionId: 'b' });
  assert.equal(closed.closed, true);
  await assert.rejects(c.command('state', { sessionId: 'b' }), { code: 'SESSION_NOT_FOUND' });
});

test('JSONL reentrant append inside awaited event acknowledgment does not deadlock', { timeout: 10000 }, async (t) => {
  let appended = false, models = 0;
  const c = client(t, async ({ method, params }, { command }) => {
    if (method === 'event') {
      if (!appended && params.event.type === 'message_end' && params.event.message.role === 'user') {
        appended = true;
        const queued = await command('append', { sessionId: 'a', additions: [user('nested arrival')] });
        assert.equal(queued.queued, true);
      }
      return { committed: true };
    }
    if (method === 'model') {
      models++;
      assert.deepEqual(params.messages.filter((item) => item.role === 'user').map((item) => item.content), ['initial', 'nested arrival']);
      return { message: { content: [{ type: 'text', text: 'done' }] } };
    }
    assert.fail('No tool callbacks expected');
  });
  await c.command('init', { sessionId: 'a', system: 'A', tools: [] });
  const result = await c.command('prompt', { sessionId: 'a', additions: [user('initial')] });
  assert.equal(models, 1);
  assert.equal(result.queuedMessages.length, 0);
});

test('JSONL abort cancels the correlated tool callback, records terminal events and ignores late result', { timeout: 10000 }, async (t) => {
  const started = deferred(), completeLate = deferred();
  let toolCallbackId, modelCount = 0;
  const c = client(t, async ({ id, method, params }) => {
    if (method === 'event') return { committed: true };
    if (method === 'model') {
      modelCount++;
      return { message: { content: [
        { type: 'toolCall', id: 'first', name: 'send', arguments: {} },
        { type: 'toolCall', id: 'second', name: 'send', arguments: {} },
      ] } };
    }
    assert.equal(method, 'tool');
    toolCallbackId = id;
    started.resolve();
    await completeLate.promise;
    return { content: [{ type: 'text', text: 'late success; host must reconcile independently' }] };
  });
  await c.command('init', { sessionId: 'a', system: 'A', tools: [tool('send')] });
  const running = c.command('prompt', { sessionId: 'a', additions: [user('initial')] });
  await started.promise;
  const end = await c.command('abort', { sessionId: 'a' });
  await running;
  assert.equal(modelCount, 1);
  assert.equal(end.paused, true);
  assert.ok(c.notifications.some((message) => message.method === 'cancel' && message.params.callbackId === toolCallbackId));
  assert.ok(c.received.some((message) => message.method === 'event' && message.params.event.type === 'run_aborted'));
  completeLate.resolve();
  const nextState = await c.command('state', { sessionId: 'a' });
  assert.deepEqual(nextState.messages, end.messages);
  await assert.rejects(c.command('continue', { sessionId: 'a' }), { code: 'UNRESOLVED_TOOL_CALLS' });
});
