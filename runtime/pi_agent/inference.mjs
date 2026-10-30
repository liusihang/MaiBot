/** Single-request services hosted by Pi ModelRuntime; never a Python model callback.
 *
 * Unlike session.*, these endpoints deliberately do not execute returned tool
 * calls. Native signed messages stay in this process behind scoped handles.
 */
import { createHash, randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SessionManager } from '@earendil-works/pi-coding-agent';
import { createIsolatedSession } from './harness.mjs';
import { createHostModelRuntime, resolveTaskModel } from './provider.mjs';

const clone = (value) => structuredClone(value);
const fail = (code, message) => Object.assign(new Error(message), { code });
const digest = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const METHODS = new Set(['model.generate', 'model.decide', 'model.embed_text', 'model.embed_image', 'model.transcribe', 'model.generate_image']);

function toolDefinitions(raw) {
  if (!Array.isArray(raw)) throw fail('INVALID_TOOLS', 'toolOptions must be an array');
  const names = new Set();
  return raw.map((entry) => {
    const tool = entry.function ?? entry;
    const parameters = tool.parameters ?? tool.parameters_schema;
    if (typeof tool.name !== 'string' || !tool.name || names.has(tool.name)
        || !parameters || typeof parameters !== 'object' || Array.isArray(parameters)) {
      throw fail('INVALID_TOOLS', 'Each tool needs a unique name and an object parameter schema');
    }
    names.add(tool.name);
    return { name: tool.name, description: String(tool.description ?? ''), parameters: clone(parameters) };
  });
}

/** Explicit wire-format mapping; unsupported providers do not silently lose JSON constraints. */
function structuredOutputHook(format, api) {
  if (format == null || format.type === 'text') return undefined;
  if (!['json_object', 'json_schema'].includes(format.type)) throw fail('UNSUPPORTED_RESPONSE_FORMAT', 'Unsupported response format');
  if (!['openai-completions', 'openai-responses'].includes(api)) {
    throw fail('UNSUPPORTED_RESPONSE_FORMAT', 'Structured output is not configured for this Pi provider');
  }
  let normalized;
  if (format.type === 'json_schema') {
    const schema = format.json_schema ?? format;
    if (typeof schema.name !== 'string' || !schema.name || !schema.schema || typeof schema.schema !== 'object') {
      throw fail('INVALID_RESPONSE_FORMAT', 'JSON schema output requires a name and schema');
    }
    normalized = { type: 'json_schema', name: schema.name, schema: clone(schema.schema),
      ...(schema.strict === undefined ? {} : { strict: schema.strict }),
      ...(schema.description === undefined ? {} : { description: schema.description }) };
  } else normalized = { type: 'json_object' };
  return (payload) => api === 'openai-responses'
    ? { ...payload, text: { ...payload.text, format: normalized } }
    : { ...payload, response_format: normalized.type === 'json_schema'
      ? { type: 'json_schema', json_schema: Object.fromEntries(Object.entries(normalized).filter(([key]) => key !== 'type')) }
      : normalized };
}

function publicMessage(message) {
  if (!message || message.role !== 'assistant' || !Array.isArray(message.content)
      || !['stop', 'toolUse', 'length', 'error', 'aborted'].includes(message.stopReason)) {
    throw fail('INVALID_PROVIDER_RESPONSE', 'Pi did not return a complete typed AssistantMessage');
  }
  const content = message.content.map((block) => {
    if (block.type === 'text') return { type: 'text', text: block.text };
    if (block.type === 'thinking') return { type: 'thinking', thinking: block.thinking };
    if (block.type === 'toolCall') return { type: 'toolCall', id: block.id, name: block.name, arguments: clone(block.arguments) };
    throw fail('UNSUPPORTED_PROVIDER_CONTENT', 'Unknown Pi assistant content type cannot be projected');
  });
  return { role: 'assistant', content, api: message.api, provider: message.provider, model: message.model,
    usage: clone(message.usage), stopReason: message.stopReason, timestamp: message.timestamp,
    ...(message.errorMessage ? { errorMessage: 'Pi provider request did not complete successfully' } : {}) };
}

/** One-shot service calls still use the full official AgentSession lifecycle.
 * Tool schemas are proposals only: the SDK records an honest policy block, not
 * a fabricated execution. A response handle refers to the official manager's
 * assistant entry, never to a second native-message ledger.
 */
export function createInferenceDispatcher({ createRuntime = createHostModelRuntime,
  resolveModel = resolveTaskModel, auxiliary, maxRecords = Number.POSITIVE_INFINITY, onSession } = {}) {
  const runtimes = new Map();
  const responses = new Map();
  const operations = new Map();
  const active = new Set();
  const root = mkdtempSync(join(tmpdir(), 'maibot-pi-services-'));
  let closed = false;
  let shutdownPromise;

  async function runtimeFor(config) {
    const key = digest(config);
    if (!runtimes.has(key)) {
      if (runtimes.size >= 8) runtimes.delete(runtimes.keys().next().value);
      const promise = createRuntime(config);
      runtimes.set(key, promise);
      promise.catch(() => { if (runtimes.get(key) === promise) runtimes.delete(key); });
    }
    return runtimes.get(key);
  }

  function resolveMessages(raw, owner, route) {
    if (!Array.isArray(raw)) throw fail('INVALID_MESSAGES', 'messages must be an array');
    const resolved = raw.flatMap((message) => {
      if (message?.role === 'assistantRef') {
        const saved = responses.get(message.handle);
        const manager = saved ? SessionManager.open(saved.file, join(root, 'sessions'), root) : undefined;
        const entry = manager?.getEntry(saved.entryId);
        if (!saved || saved.owner !== owner || saved.routeFingerprint !== route.routeFingerprint
            || entry?.type !== 'message' || entry.message.role !== 'assistant' || digest(entry.message) !== saved.nativeHash) {
          throw fail('NATIVE_REFERENCE_UNAVAILABLE', 'Native assistant reference is unavailable in this session and route');
        }
        return clone(entry.message);
      }
      if (message?.role === 'externalToolExchange') {
        if (message.provenance !== 'caller_public_history'
            || Object.keys(message).some((key) => !['role', 'provenance', 'calls', 'results', 'timestamp'].includes(key))
            || !Array.isArray(message.calls) || !message.calls.length || !Array.isArray(message.results)
            || message.calls.length !== message.results.length) {
          throw fail('INVALID_EXTERNAL_TOOL_EXCHANGE', 'External tool history must be explicit, public and fully paired');
        }
        const ids = new Set();
        const calls = message.calls.map((call) => {
          if (Object.keys(call).some((key) => !['id', 'name', 'arguments'].includes(key))
              || typeof call.id !== 'string' || !call.id || ids.has(call.id)
              || typeof call.name !== 'string' || !call.name || call.arguments == null
              || typeof call.arguments !== 'object' || Array.isArray(call.arguments)) {
            throw fail('INVALID_EXTERNAL_TOOL_EXCHANGE', 'External tool call has invalid public fields');
          }
          ids.add(call.id);
          return { type: 'toolCall', ...clone(call) };
        });
        const results = message.results.map((result) => {
          const call = calls.find((candidate) => candidate.id === result.toolCallId);
          if (!call || !ids.delete(result.toolCallId) || result.toolName !== call.name
              || Object.keys(result).some((key) => !['toolCallId', 'toolName', 'content', 'isError', 'timestamp'].includes(key))
              || !Array.isArray(result.content) || result.content.some((part) =>
                part?.type === 'text' ? typeof part.text !== 'string' || Object.keys(part).some((key) => !['type', 'text'].includes(key))
                  : part?.type === 'image' ? typeof part.data !== 'string' || typeof part.mimeType !== 'string'
                    || Object.keys(part).some((key) => !['type', 'data', 'mimeType'].includes(key)) : true)) {
            throw fail('INVALID_EXTERNAL_TOOL_EXCHANGE', 'External tool result has invalid public fields or pair identity');
          }
          return { role: 'toolResult', ...clone(result), isError: result.isError === true, timestamp: result.timestamp ?? 0 };
        });
        return [{ role: 'assistant', content: calls, api: route.model.api, provider: route.model.provider,
          model: route.model.id, timestamp: message.timestamp ?? 0, stopReason: 'toolUse',
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } }, ...results];
      }
      if (message?.role === 'assistantExample') {
        const parts = typeof message.content === 'string' ? [{ type: 'text', text: message.content }] : message.content;
        if (!Array.isArray(parts) || parts.some((part) => part?.type !== 'text'
            || typeof part.text !== 'string' || Object.keys(part).some((key) => !['type', 'text'].includes(key)))
            || Object.keys(message).some((key) => !['role', 'content', 'timestamp'].includes(key))) {
          throw fail('INVALID_ASSISTANT_EXAMPLE', 'Portable assistant examples allow plain text only, not native state, images, or tool calls');
        }
        return { role: 'assistant', content: clone(parts), api: route.model.api, provider: route.model.provider,
          model: route.model.id, timestamp: message.timestamp ?? 0, stopReason: 'stop',
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
      }
      if (!message || !['system', 'user', 'toolResult'].includes(message.role)) {
        throw fail('UNTRUSTED_ASSISTANT_REPLAY', 'Assistant history must use a Node-owned assistantRef or explicit portable assistantExample');
      }
      if (typeof message.content !== 'string' && !Array.isArray(message.content)) {
        throw fail('INVALID_MESSAGES', 'Message content must be text or content blocks');
      }
      // Inputs are public projections. Reject attempts to smuggle provider-native
      // or tool-declaration fields through a permissive object copy.
      const allowed = message.role === 'toolResult'
        ? ['role', 'content', 'timestamp', 'toolCallId', 'toolName', 'isError', 'details']
        : ['role', 'content', 'timestamp'];
      if (Object.keys(message).some((key) => !allowed.includes(key))) throw fail('INVALID_MESSAGES', 'Unexpected native or control field in public message');
      const copy = clone(message);
      copy.timestamp ??= 0;
      return copy;
    });
    const pending = new Map();
    const seen = new Set();
    for (const message of resolved) {
      if (message.role === 'assistant') {
        if (pending.size) throw fail('UNRESOLVED_TOOL_CALLS', 'Every previous tool proposal requires a caller-supplied result before another assistant');
        for (const block of message.content) if (block.type === 'toolCall') {
          if (seen.has(block.id)) throw fail('INVALID_TOOL_PAIR', 'Duplicate tool call ID in auxiliary history');
          seen.add(block.id); pending.set(block.id, block.name);
        }
      } else if (message.role === 'toolResult') {
        if (pending.get(message.toolCallId) !== message.toolName) throw fail('INVALID_TOOL_PAIR', 'Tool result has no matching Node-owned assistant call');
        pending.delete(message.toolCallId);
      } else if (pending.size) throw fail('UNRESOLVED_TOOL_CALLS', 'Tool results must immediately follow their assistant calls');
    }
    if (pending.size) throw fail('UNRESOLVED_TOOL_CALLS', 'All auxiliary tool proposals must be resolved before the next inference');
    return resolved;
  }

  async function execute(method, params, signal) {
    signal.throwIfAborted();
    if (closed) throw fail('INFERENCE_CLOSED', 'Inference dispatcher is closed');
    if (!params || typeof params !== 'object' || typeof params.taskName !== 'string') {
      throw fail('INVALID_INFERENCE_REQUEST', 'A configured taskName is required');
    }
    const runtime = await runtimeFor(params.config);
    signal.throwIfAborted();
    if (method === 'model.decide') {
      return (await import('./decisions.mjs')).decideParticipation(runtime, params, { signal });
    }
    if (method !== 'model.generate') {
      const run = auxiliary ?? (await import('./auxiliary.mjs')).runAuxiliaryOperation;
      return run(runtime, method.slice('model.'.length), params, { signal });
    }
    const owner = String(params.sessionId ?? '');
    const options = params.options ?? {};
    const retainReplay = options.retainReplay !== false;
    if (retainReplay && responses.size >= maxRecords) throw fail('INFERENCE_CAPACITY', 'Native response store is full; release completed references');
    const route = resolveModel(runtime, params.taskName, { modelOverride: params.modelOverride, options });
    const messages = resolveMessages(params.messages, owner, route);
    const tools = toolDefinitions(options.toolOptions ?? options.tools ?? []);
    const onPayload = structuredOutputHook(options.responseFormat ?? options.response_format, route.model.api);
    const signals = [signal];
    if (Number.isFinite(route.timeoutMs) && route.timeoutMs > 0) signals.push(AbortSignal.timeout(Math.ceil(route.timeoutMs)));
    const requestSignal = AbortSignal.any(signals);
    requestSignal.throwIfAborted();
    let leadingSystems = 0;
    while (messages[leadingSystems]?.role === 'system') leadingSystems++;
    if (messages.slice(leadingSystems).some((message) => message.role === 'system')) {
      throw fail('INVALID_SYSTEM_POSITION', 'Auxiliary system instructions must precede public conversation content');
    }
    const system = messages.slice(0, leadingSystems).map((message) => {
      if (typeof message.content !== 'string') throw fail('INVALID_SYSTEM_CONTENT', 'System instructions must be plain text');
      return message.content;
    }).join('\n\n');
    const manager = SessionManager.create(root, join(root, 'sessions'));
    // SessionManager is authoritative even for short-lived jobs. Its disk
    // archive is lazily reopened by handles, not retained as an unbounded
    // collection of full in-memory transcripts. No native
    // assistant bytes return to Python or get reconstituted from its projection.
    for (const message of messages.slice(leadingSystems)) manager.appendMessage(message);
    let session;
    let retained = false;
    try {
      session = await createIsolatedSession({ cwd: join(root, 'cwd'), agentDir: join(root, 'agent'),
        modelRuntime: runtime, resolved: route, sessionManager: manager, system,
        customTools: tools.map((tool) => ({ ...tool, label: tool.name,
          execute: async () => { throw fail('PROPOSAL_EXECUTION_FORBIDDEN', 'Proposal tools must never execute'); } })),
        extensionFactory: (pi) => {
          pi.on('tool_call', () => ({ block: true, terminate: true,
            reason: 'Proposal-only model service: tool was not executed; the requesting caller must authorize and execute it.' }));
          if (onPayload) pi.on('before_provider_request', ({ payload }) => onPayload(payload));
        },
      });
      const finish = session.agent.finishTurn;
      session.agent.finishTurn = async (turn, currentSignal) => {
        await finish?.(turn, currentSignal);
        return { action: 'end' };
      };
      const abort = () => { void session.abort(); };
      requestSignal.addEventListener('abort', abort, { once: true });
      try {
        requestSignal.throwIfAborted();
        onSession?.(session, manager);
        const previousIds = new Set(manager.getEntries().map((entry) => entry.id));
        // Explicit host activation, not an invented repeated user utterance.
        await session.sendCustomMessage({ customType: 'maibot.service_request',
          content: 'Process the supplied request once. Any tool calls are proposals only; do not execute tools.',
          display: false }, { triggerTurn: true });
        requestSignal.throwIfAborted();
        const entry = manager.getEntries().find((candidate) => !previousIds.has(candidate.id)
          && candidate.type === 'message' && candidate.message.role === 'assistant');
        if (!entry) throw fail('INVALID_PROVIDER_RESPONSE', 'Pi did not persist an assistant response');
        const message = entry.message;
        if (['error', 'aborted'].includes(message.stopReason)
            || (message.stopReason === 'length' && message.content.some((part) => part.type === 'toolCall'))) {
          const code = message.stopReason === 'length' ? 'TRUNCATED_TOOL_PROPOSAL'
            : message.stopReason === 'aborted' ? 'MODEL_ABORTED' : 'MODEL_REQUEST_FAILED';
          const failure = fail(code, message.stopReason === 'length'
            ? 'Pi response reached its output limit; incomplete tool proposals are not executable'
            : 'Pi provider request did not complete successfully');
          failure.safeDiagnostics = { route: { modelName: route.hostModelName, modelIdentifier: route.model.id,
            apiProvider: route.apiProvider, routeFingerprint: route.routeFingerprint },
            usage: clone(message.usage), stopReason: message.stopReason,
            ...(typeof message.durationMs === 'number' ? { durationMs: message.durationMs } : {}) };
          throw failure;
        }
        const projection = publicMessage(message);
        const responseHandle = retainReplay ? `pi-response:${randomUUID()}` : undefined;
        if (retainReplay) {
          responses.set(responseHandle, { owner, routeFingerprint: route.routeFingerprint, file: manager.getSessionFile(), entryId: entry.id, nativeHash: digest(message) });
          retained = true;
        }
        return { message: projection, ...(responseHandle ? { responseHandle } : {}),
          replayRetained: retained, usage: clone(message.usage), execution: 'proposal_only',
          effectiveOptions: { ...route.requestOptions, ...(options.responseFormat ? { responseFormat: clone(options.responseFormat) } : {}) },
          route: { modelName: route.hostModelName, modelIdentifier: route.model.id,
            apiProvider: route.apiProvider, routeFingerprint: route.routeFingerprint },
        };
      } finally { requestSignal.removeEventListener('abort', abort); }
    } finally {
      session?.dispose();
      // Explicit one-shot jobs and failed calls have no live replay handle.
      // Remove their private prompt/archive immediately, not only at shutdown.
      if (!retained && manager.getSessionFile()) rmSync(manager.getSessionFile(), { force: true });
    }
  }

  function keyFor(params) { return `${String(params.sessionId ?? '')}\0${params.operationId}`; }
  return {
    async handle(method, params = {}, { signal } = {}) {
      if (method === 'model.cancel') {
        const pending = operations.get(keyFor(params));
        if (!pending || pending.settled) return { cancelled: false };
        pending.controller.abort();
        return { cancelled: true };
      }
      if (method === 'model.release') {
        let released = 0;
        for (const handle of params.handles ?? []) {
          const saved = responses.get(handle);
          if (saved?.owner === String(params.sessionId ?? '') && responses.delete(handle)) {
            rmSync(saved.file, { force: true }); released += 1;
          }
        }
        // The caller explicitly retires dedup IDs only after it will no longer
        // retry them. Active operations cannot be forgotten or duplicated.
        for (const operationId of params.operationIds ?? []) {
          const key = keyFor({ ...params, operationId });
          if (operations.get(key)?.settled) operations.delete(key);
        }
        return { released };
      }
      if (closed) throw fail('INFERENCE_CLOSED', 'Inference dispatcher is closed');
      if (!METHODS.has(method)) throw fail('UNKNOWN_INFERENCE_OPERATION', `Unsupported inference operation: ${method}`);
      signal?.throwIfAborted();
      const key = params.operationId ? keyFor(params) : null;
      const hash = digest({ method, params });
      const previous = key && operations.get(key);
      if (previous) {
        if (previous.hash !== hash) throw fail('OPERATION_ID_CONFLICT', 'Operation ID was reused with different input');
        return previous.promise.then(clone);
      }
      if (key && operations.size >= maxRecords) throw fail('INFERENCE_CAPACITY', 'Inference operation store is full; release completed operation IDs');
      const controller = new AbortController();
      const abort = () => controller.abort(signal?.reason);
      signal?.addEventListener('abort', abort, { once: true });
      const record = { hash, controller, settled: false };
      active.add(record);
      record.promise = execute(method, params, controller.signal).finally(() => {
        record.settled = true; active.delete(record); signal?.removeEventListener('abort', abort);
      });
      if (key) operations.set(key, record);
      return record.promise.then(clone);
    },
    close() {
      if (shutdownPromise) return shutdownPromise;
      closed = true;
      for (const record of active) record.controller.abort();
      responses.clear(); operations.clear(); runtimes.clear();
      // Abort completion still writes its terminal SDK entry. Do not unlink the
      // archive underneath that awaited lifecycle.
      shutdownPromise = Promise.allSettled([...active].map((record) => record.promise))
        .then(() => { rmSync(root, { recursive: true, force: true }); });
      return shutdownPromise;
    },
  };
}
