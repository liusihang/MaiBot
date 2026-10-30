import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { answer, call, fixtureConfig, fixtureResponse, reasoning } from './helpers/mock-provider.mjs';

const sidecar = fileURLToPath(new URL('../sidecar.mjs', import.meta.url));
const user = (content) => ({ role: 'user', content, timestamp: 1, hostEventId: content });
const tool = (name) => ({ name, description: name, parameters: { type: 'object' } });
const deferred = () => { let resolve; const promise = new Promise((yes) => { resolve = yes; }); return { promise, resolve }; };

/** Real subprocess, SDK and localhost official Responses wire; no model callback exists. */
async function client(t, callback, respond) {
  const root = mkdtempSync(join(tmpdir(), 'pi-stdio-'));
  const wires = [];
  const server = createServer(async (req, res) => {
    try {
      const chunks = []; for await (const chunk of req) chunks.push(chunk);
      const wire = JSON.parse(Buffer.concat(chunks).toString()); wires.push(wire);
      const response = fixtureResponse(await respond(wire), `response-${wires.length}`);
      res.writeHead(200, { 'content-type': 'text/event-stream' }); res.end(await response.text());
    } catch (error) { res.writeHead(500); res.end(String(error)); }
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const config = fixtureConfig(); config.api_providers[0].base_url = `http://127.0.0.1:${server.address().port}/v1`;
  const child = spawn(process.execPath, [sidecar], { stdio: ['pipe', 'pipe', 'pipe'] });
  const pending = new Map(), notifications = [], received = [], callbackErrors = [];
  let nextId = 0, stderr = '';
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  const command = (method, params) => {
    const id = `request-${++nextId}`;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      send({ id, method, params: method === 'init' ? { sessionDir: root, config, ...params } : params });
    });
  };
  const lineReader = createInterface({ input: child.stdout });
  lineReader.on('line', (line) => {
    const message = JSON.parse(line); received.push(message);
    if (message.method) {
      if (!message.id) { notifications.push(message); return; }
      void (async () => {
        try {
          const result = message.method === 'authorize' ? { allowed: true } : await callback(message, { command, send, child });
          send({ id: message.id, result });
        } catch (error) {
          callbackErrors.push(error); send({ id: message.id, error: { code: 'MOCK_ERROR', message: error.message } });
        }
      })();
      return;
    }
    const item = pending.get(message.id); if (!item) return;
    pending.delete(message.id);
    if (message.error) item.reject(Object.assign(new Error(message.error.message), message.error)); else item.resolve(message.result);
  });
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) { child.stdin.end(); await once(child, 'exit'); }
    server.closeAllConnections(); await new Promise((done) => server.close(done));
    rmSync(root, { recursive: true, force: true });
    assert.equal(stderr, ''); assert.deepEqual(callbackErrors, []);
  });
  return { child, command, send, notifications, received, wires };
}

test('JSONL subprocess routes concurrent groups and out-of-order callbacks while preserving native replay only in Node', { timeout: 20000 }, async (t) => {
  const firstStarted = deferred(), secondStarted = deferred(), releaseFirst = deferred();
  const counts = new Map();
  const c = await client(t, async (message) => {
    assert.equal(message.method, 'event'); return {};
  }, async (wire) => {
    const sid = wire.input[0].content.toLowerCase();
    counts.set(sid, (counts.get(sid) ?? 0) + 1);
    if (sid === 'a' && counts.get(sid) === 1) { firstStarted.resolve(); await releaseFirst.promise; }
    else if (sid === 'b') secondStarted.resolve();
    return [reasoning(), answer(sid, `message-${sid}-${counts.get(sid)}`)];
  });
  await c.command('init', { sessionId: 'a', system: 'A', tools: [] });
  await c.command('init', { sessionId: 'b', system: 'B', tools: [] });
  const first = c.command('prompt', { sessionId: 'a', additions: [user('a input')] });
  await firstStarted.promise;
  const second = c.command('prompt', { sessionId: 'b', additions: [user('b input')] });
  await secondStarted.promise; assert.equal((await second).conclusion, 'b');
  releaseFirst.resolve(); const a = await first;
  assert.equal(a.conclusion, 'a'); assert.equal(a.messages, undefined);
  assert.equal(JSON.stringify(c.received).includes('opaque-encrypted'), false);
  await c.command('prompt', { sessionId: 'a', additions: [user('another a')] });
  assert.ok(JSON.stringify(c.wires.at(-1)).includes('opaque-encrypted-preserve-exactly'));
  assert.deepEqual(c.wires.at(-1).input.slice(0, c.wires[0].input.length), c.wires[0].input);
  assert.equal(c.wires.filter((wire) => wire.input[0].content === 'A').some((wire) => JSON.stringify(wire).includes('b input')), false);
  assert.equal((await c.command('close', { sessionId: 'b' })).closed, true);
  await assert.rejects(c.command('state', { sessionId: 'b' }), { code: 'UNKNOWN_SESSION' });
});

test('JSONL reentrant append inside awaited public event notification does not deadlock or duplicate source text', { timeout: 20000 }, async (t) => {
  let appended = false;
  const c = await client(t, async ({ method, params }, { command }) => {
    assert.equal(method, 'event');
    if (!appended && params.event.type === 'agent_start') {
      appended = true;
      const queued = await command('append', { sessionId: 'a', additions: [user('nested arrival')] });
      assert.equal(queued.queued, true); assert.deepEqual(queued.acceptedSourceIds, ['nested arrival']);
    }
    return {};
  }, async (wire) => {
    const text = JSON.stringify(wire);
    assert.ok(text.includes('initial')); assert.equal(text.match(/nested arrival/g)?.length, 1);
    return [answer('done')];
  });
  await c.command('init', { sessionId: 'a', system: 'A', tools: [] });
  const result = await c.command('prompt', { sessionId: 'a', additions: [user('initial')] });
  assert.equal(c.wires.length, 1); assert.equal(result.queuedMessageCount, 0);
});

test('JSONL abort cancels correlated business tool callback, records terminal evidence and ignores late result', { timeout: 20000 }, async (t) => {
  const started = deferred(), completeLate = deferred(); let toolCallbackId;
  const c = await client(t, async ({ id, method }) => {
    if (method === 'event') return {};
    assert.equal(method, 'tool'); toolCallbackId = id; started.resolve(); await completeLate.promise;
    return { content: [{ type: 'text', text: 'late success; reconcile independently' }] };
  }, async () => [call('send', {}, 'first', 'f1'), call('send', {}, 'second', 'f2')]);
  await c.command('init', { sessionId: 'a', system: 'A', tools: [tool('send')] });
  const running = c.command('prompt', { sessionId: 'a', additions: [user('initial')] });
  await started.promise;
  const end = await c.command('abort', { sessionId: 'a' }); await running;
  assert.equal(c.wires.length, 1); assert.equal(end.paused, true);
  assert.ok(c.notifications.some((message) => message.method === 'cancel' && message.params.callbackId === toolCallbackId));
  assert.ok(c.received.some((message) => message.method === 'event' && message.params.event.type === 'message_end'
    && message.params.event.message.role === 'toolResult' && message.params.event.message.isError));
  completeLate.resolve(); const next = await c.command('state', { sessionId: 'a' });
  assert.deepEqual(next.unresolvedToolCalls, end.unresolvedToolCalls); assert.equal(next.revision, end.revision);
  await assert.rejects(c.command('continue', { sessionId: 'a' }), { code: 'UNRESOLVED_TOOL_CALLS' });
});
