import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { PassThrough } from 'node:stream';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { StdioBridge } from '../sidecar.mjs';
import { fixtureConfig } from './helpers/mock-provider.mjs';

const sidecar = fileURLToPath(new URL('../sidecar.mjs', import.meta.url));
const delay = (ms) => new Promise((done) => setTimeout(done, ms));

function processBridge(t) {
  const child = spawn(process.execPath, [sidecar], { stdio: ['pipe', 'pipe', 'pipe'] });
  const pending = new Map(); let sequence = 0, diagnostics = '';
  child.stderr.on('data', (data) => { diagnostics += data; });
  const lines = createInterface({ input: child.stdout });
  lines.on('line', (line) => {
    const message = JSON.parse(line);
    if (message.method && message.id) {
      child.stdin.write(JSON.stringify({ id: message.id, result: message.method === 'authorize' ? { allowed: true } : {} }) + '\n');
      return;
    }
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id); clearTimeout(request.timeout);
    if (message.error) request.reject(Object.assign(new Error(message.error.message), { code: message.error.code }));
    else request.resolve(message.result);
  });
  child.on('exit', () => {
    for (const request of pending.values()) { clearTimeout(request.timeout); request.reject(new Error(`sidecar exited: ${diagnostics}`)); }
    pending.clear();
  });
  const stop = async (signal = 'SIGTERM') => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const exited = once(child, 'exit'); child.kill(signal); await exited;
  };
  t.after(() => stop());
  return { child, stop, request(method, params) {
    return new Promise((resolve, reject) => {
      const id = `test-${++sequence}`;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method} ${diagnostics}`)); }, 10000);
      pending.set(id, { resolve, reject, timeout });
      child.stdin.write(JSON.stringify({ id, method, params }) + '\n');
    });
  } };
}

test('real separate sidecars exclude simultaneous writers and recover persisted input after SIGKILL', async (t) => {
  const root = mkdtempSync(join(tmpdir(), 'pi-lock-process-'));
  const a = processBridge(t), b = processBridge(t);
  t.after(async () => { await Promise.all([a.stop(), b.stop()]); rmSync(root, { recursive: true, force: true }); });
  const params = { sessionId: 'real-group', sessionDir: root, system: 'persona', tools: [], config: fixtureConfig(), taskName: 'planner' };
  await a.request('init', params);
  await assert.rejects(b.request('init', params), { code: 'SESSION_LOCKED' });
  const addition = { role: 'user', content: 'durable despite process death', timestamp: 1, hostEventId: 'source-1' };
  assert.deepEqual((await a.request('append', { sessionId: params.sessionId, additions: [addition] })).acceptedSourceIds, ['source-1']);
  await a.stop('SIGKILL');
  const deadline = Date.now() + 5000;
  for (;;) {
    try { await b.request('init', params); break; }
    catch (error) { if (error.code !== 'SESSION_LOCKED' || Date.now() >= deadline) throw error; await delay(20); }
  }
  const resumed = await b.request('append', { sessionId: params.sessionId, additions: [addition] });
  assert.equal(resumed.messages, undefined);
  const archive = await b.request('read_archive', { sessionId: params.sessionId, eventIds: resumed.archive.recentEventIds, limit: 12000 });
  assert.equal(archive.text.match(/durable despite process death/g)?.length, 1);
  await b.request('close', { sessionId: params.sessionId });
  const c = processBridge(t); await c.request('init', params); await c.stop();
});

test('sidecar routes all model capabilities to dispatcher and close aborts in-flight operation signals', async () => {
  const input = new PassThrough(), output = new PassThrough();
  const replies = [], methods = []; let pendingSignal, resolveStarted;
  const started = new Promise((resolve) => { resolveStarted = resolve; });
  output.on('data', (data) => replies.push(...data.toString().trim().split('\n').filter(Boolean).map(JSON.parse)));
  const runtime = { handle: async (method) => ({ business: method }), shutdown: async () => {} };
  const inference = { handle: async (method, params, { signal }) => {
    methods.push(method); assert.ok(signal instanceof AbortSignal);
    if (params.wait) { pendingSignal = signal; resolveStarted(); await new Promise((_yes, no) => signal.addEventListener('abort', () => no(new DOMException('aborted', 'AbortError')), { once: true })); }
    return { model: method };
  }, close() {} };
  const bridge = new StdioBridge({ input, output, runtime, inference });
  for (const method of ['model.generate', 'model.embed_text', 'model.embed_image', 'model.transcribe', 'model.generate_image', 'model.release', 'model.cancel']) {
    await bridge.receive({ id: method, method, params: {} });
    assert.equal(replies.at(-1).result.model, method);
  }
  await bridge.receive({ id: 'state', method: 'state', params: {} });
  assert.equal(replies.at(-1).result.business, 'state');
  const running = bridge.receive({ id: 'long', method: 'model.generate', params: { wait: true } });
  await started; await bridge.close(); await running;
  assert.equal(pendingSignal.aborted, true);
  assert.equal(bridge.operations.size, 0);
  assert.equal(methods.length, 8);
});

test('duplex callback IDs settle out of order and cancelled callbacks ignore late replies', async () => {
  const output = new PassThrough(), sent = [];
  output.on('data', (data) => sent.push(...data.toString().trim().split('\n').filter(Boolean).map(JSON.parse)));
  const bridge = new StdioBridge({ input: new PassThrough(), output,
    runtime: { shutdown: async () => {} }, inference: { close() {} } });
  const a = bridge.callback('event', { sessionId: 'a' });
  const control = new AbortController(); const b = bridge.callback('tool', { sessionId: 'a' }, control.signal);
  const rejection = assert.rejects(b, { name: 'AbortError' });
  control.abort(); await rejection;
  await bridge.receive({ id: sent[1].id, result: { late: true } });
  await bridge.receive({ id: sent[0].id, result: { recorded: true } });
  assert.deepEqual(await a, { recorded: true });
  assert.equal(bridge.pending.size, 0);
  await bridge.close();
});


test('JSONL errors expose only explicit safeDiagnostics, never arbitrary provider details', async () => {
  const output = new PassThrough(), replies = [];
  output.on('data', (data) => replies.push(JSON.parse(data.toString())));
  const bridge = new StdioBridge({ input: new PassThrough(), output,
    runtime: { shutdown: async () => {} }, inference: {
      handle: async () => { throw Object.assign(new Error('provider request failed'), {
        code: 'PROVIDER_ERROR', safeDiagnostics: { stopReason: 'error', durationMs: 12 },
        details: { secret: 'private' }, response: { native: 'opaque' },
      }); }, close() {},
    } });
  await bridge.receive({ id: 'safe', method: 'model.generate', params: {} });
  assert.deepEqual(replies[0].error.details, { stopReason: 'error', durationMs: 12 });
  assert.equal(JSON.stringify(replies).includes('private'), false);
  assert.equal(JSON.stringify(replies).includes('opaque'), false);
  await bridge.close();
});
