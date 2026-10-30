import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { PiRuntime, RuntimeError } from './runtime.mjs';
import { createInferenceDispatcher } from './inference.mjs';

/** Duplex JSONL. Requests and callbacks use separate IDs, and callbacks may settle out of order. */
export class StdioBridge {
  constructor({ input = process.stdin, output = process.stdout, diagnostic = process.stderr, runtime, inference } = {}) {
    this.input = input;
    this.output = output;
    this.diagnostic = diagnostic;
    this.pending = new Map();
    this.requests = new Set();
    this.nextId = 0;
    this.closed = false;
    this.runtime = runtime ?? new PiRuntime((method, params, signal) => this.callback(method, params, signal));
    this.inference = inference ?? createInferenceDispatcher();
    this.operations = new Map();
  }

  send(message) {
    if (this.closed) throw new RuntimeError('TRANSPORT_CLOSED', 'Python host disconnected');
    this.output.write(`${JSON.stringify(message)}\n`);
  }

  callback(method, params, signal) {
    if (signal?.aborted) return Promise.reject(new DOMException('Operation aborted', 'AbortError'));
    return new Promise((resolve, reject) => {
      const id = `cb-${++this.nextId}`;
      const abort = () => {
        if (!this.pending.delete(id)) return;
        signal?.removeEventListener('abort', abort);
        try { this.send({ method: 'cancel', params: { callbackId: id, sessionId: params.sessionId } }); } catch {}
        reject(new DOMException('Operation aborted', 'AbortError'));
      };
      this.pending.set(id, { resolve, reject, cleanup: () => signal?.removeEventListener('abort', abort) });
      signal?.addEventListener('abort', abort, { once: true });
      try { this.send({ id, method, params }); } catch (error) {
        this.pending.delete(id);
        signal?.removeEventListener('abort', abort);
        reject(error);
      }
    });
  }

  async receive(message) {
    const { id, method } = message;
    if (typeof method !== 'string') {
      const pending = this.pending.get(id);
      // A callback canceled while Python was returning can legitimately arrive late.
      if (!pending) return;
      this.pending.delete(id);
      pending.cleanup();
      if (message.error) {
        pending.reject(new RuntimeError(message.error.code ?? 'HOST_ERROR', message.error.message ?? 'Host callback failed'));
      } else {
        pending.resolve(message.result);
      }
      return;
    }
    if ((typeof id !== 'string' && typeof id !== 'number') || String(id).startsWith('cb-')) {
      this.send({ id: id ?? null, error: { code: 'INVALID_ID', message: 'Command id is required and cb- is reserved' } });
      return;
    }
    if (this.requests.has(id)) {
      this.send({ id, error: { code: 'DUPLICATE_ID', message: 'Command id is already in flight' } });
      return;
    }
    this.requests.add(id);
    try {
      const params = message.params ?? {};
      let result;
      if (method.startsWith('model.')) {
        const controller = new AbortController();
        this.operations.set(id, controller);
        result = await this.inference.handle(method, params, { signal: controller.signal });
      } else result = await this.runtime.handle(method, params);
      this.send({ id, result });
    } catch (error) {
      if (!this.closed) this.send({ id, error: {
        code: error.code ?? 'RUNTIME_ERROR', message: error instanceof Error ? error.message : String(error),
        ...(error.safeDiagnostics ? { details: error.safeDiagnostics } : {}),
      } });
    } finally {
      this.requests.delete(id);
      this.operations.delete(id);
    }
  }

  start() {
    this.lines = createInterface({ input: this.input, crlfDelay: Infinity });
    this.lines.on('line', (line) => {
      if (!line.trim()) return;
      try {
        const message = JSON.parse(line);
        if (!message || typeof message !== 'object' || Array.isArray(message)) throw new Error('Expected a JSON object');
        // Never await a command in the input pump: event acknowledgments, new group
        // messages and abort commands must be handled while a prompt is in progress.
        void this.receive(message).catch((error) => this.diagnostic.write(`${error.stack ?? error}\n`));
      } catch (error) {
        this.send({ id: null, error: { code: 'INVALID_JSON', message: error.message } });
      }
    });
    this.lines.on('close', () => this.close());
    this.input.on('error', () => this.close());
    this.output.on('error', () => this.close());
    return this;
  }

  close() {
    if (this.closed) return this.closePromise;
    this.closed = true;
    for (const controller of this.operations.values()) controller.abort();
    // Both lifecycles may still append terminal SDK records after cancellation.
    // SIGTERM/EOF must wait before process exit or temporary archive cleanup.
    this.closePromise = Promise.allSettled([this.runtime.shutdown(), this.inference.close()]);
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(new RuntimeError('TRANSPORT_CLOSED', 'Python host disconnected'));
    }
    this.pending.clear();
    return this.closePromise;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const bridge = new StdioBridge().start();
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, async () => {
      await bridge.close();
      process.exit(0);
    });
  }
}
