import { Agent } from '@earendil-works/pi-agent-core';
import {
  createAssistantMessageEventStream,
  getCurrentSystemPrompt,
  getCurrentTools,
} from '@earendil-works/pi-ai';

const HOST_MODEL = Object.freeze({
  id: 'maibot-host', name: 'MaiBot Python provider', api: 'maibot-host',
  provider: 'maibot-host', baseUrl: '', reasoning: false, input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 0, maxTokens: 0,
});
const EMPTY_USAGE = Object.freeze({
  input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});
const ROLES = new Set(['system', 'user', 'assistant', 'toolResult']);
const STOP_REASONS = new Set(['stop', 'length', 'toolUse', 'error', 'aborted']);
const clone = (value) => structuredClone(value);
const canonical = (value) => JSON.stringify(value, (_key, item) => (
  item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]]))
    : item
));

export class RuntimeError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'RuntimeError';
    this.code = code;
  }
}

function check(condition, code, message) {
  if (!condition) throw new RuntimeError(code, message);
}

function normalizeTool(tool) {
  check(tool && typeof tool === 'object', 'INVALID_TOOL', 'Tool must be an object');
  check(typeof tool.name === 'string' && tool.name.length > 0, 'INVALID_TOOL', 'Tool name is required');
  check(typeof tool.description === 'string', 'INVALID_TOOL', 'Tool description is required');
  check(tool.parameters && typeof tool.parameters === 'object' && !Array.isArray(tool.parameters),
    'INVALID_TOOL', `Tool ${tool.name} requires JSON Schema parameters`);
  return {
    name: tool.name, description: tool.description, parameters: clone(tool.parameters),
    ...(tool.constrainedSampling === undefined ? {} : { constrainedSampling: tool.constrainedSampling }),
  };
}

function validateHistory(messages, system) {
  check(Array.isArray(messages), 'INVALID_MESSAGES', 'messages must be an array');
  const result = clone(messages);
  for (const [index, message] of result.entries()) {
    check(message && ROLES.has(message.role), 'INVALID_MESSAGES', `Invalid message role at index ${index}`);
    check(typeof message.content === 'string' || Array.isArray(message.content),
      'INVALID_MESSAGES', `Message ${index} has no valid content`);
    if (message.role === 'system') {
      check(index === 0 || message.content === '' || (Array.isArray(message.content) && message.content.length === 0),
        'SYSTEM_IMMUTABLE', 'Only the leading system message may contain instructions');
      check(!message.sections && !message.toolsRemoved?.length,
        'SYSTEM_IMMUTABLE', 'Prompt-section replacement and tool removal are not supported');
    }
  }
  if (result[0]?.role === 'system') {
    check(getCurrentSystemPrompt(result) === system, 'SYSTEM_IMMUTABLE', 'Replayed base system prompt differs from init.system');
  }
  const definitions = new Map();
  for (const message of result) {
    for (const tool of message.toolsAdded ?? []) {
      const normalized = normalizeTool(tool);
      const previous = definitions.get(tool.name);
      check(!previous || canonical(previous) === canonical(normalized),
        'TOOL_IMMUTABLE', `History redefines tool ${tool.name}`);
      definitions.set(tool.name, normalized);
    }
  }
  return result;
}

function normalizeAdditions(additions) {
  check(Array.isArray(additions) && additions.length > 0,
    'INVALID_MESSAGES', 'additions must be a non-empty array of user messages');
  return additions.map((message) => {
    check(message?.role === 'user', 'INVALID_MESSAGES', 'External additions must have role user');
    check(typeof message.content === 'string' || Array.isArray(message.content),
      'INVALID_MESSAGES', 'External message content must be a string or array');
    return { ...clone(message), timestamp: message.timestamp ?? Date.now() };
  });
}

function normalizeAssistant(result, model) {
  // A host admission decision is not provider output. Pi requires a terminal
  // stream message, so retain an explicitly labeled, non-provider control receipt
  // in the audit transcript; native.items=[] contributes nothing to model replay.
  if (result?.maintenanceRequired === true && result.message === undefined) {
    check(result.hostControl === 'capacity_admission', 'INVALID_HOST_CONTROL',
      'Maintenance sentinel must explicitly identify capacity_admission');
    check(typeof result.reason === 'string' && result.reason.length > 0,
      'INVALID_HOST_CONTROL', 'Capacity admission control requires a reason');
    result = {
      role: 'assistant', content: [], stopReason: 'stop', native: { items: [] },
      hostControl: 'capacity_admission', hostControlReason: result.reason,
      hostMaintenanceRequired: true, hostContinuationRequired: true,
    };
  }
  const value = result?.message ?? result;
  check(value && typeof value === 'object' && Array.isArray(value.content),
    'INVALID_MODEL_RESULT', 'Model callback must return {message: {content: [...]}}');
  check(value.role === undefined || value.role === 'assistant', 'INVALID_MODEL_RESULT', 'Model result role must be assistant');
  const message = {
    ...clone(value), role: 'assistant', content: clone(value.content),
    api: value.api ?? model.api, provider: value.provider ?? model.provider,
    model: value.model ?? model.id, usage: value.usage ?? clone(EMPTY_USAGE),
    stopReason: value.stopReason ?? (value.content.some((part) => part.type === 'toolCall') ? 'toolUse' : 'stop'),
    timestamp: value.timestamp ?? Date.now(),
  };
  check(STOP_REASONS.has(message.stopReason), 'INVALID_MODEL_RESULT', `Invalid stop reason ${message.stopReason}`);
  for (const part of message.content) {
    check(part && ['text', 'thinking', 'toolCall', 'redactedThinking'].includes(part.type),
      'INVALID_MODEL_RESULT', 'Unknown assistant content block; keep provider-specific blocks in native');
    if (part.type === 'toolCall') {
      check(typeof part.id === 'string' && typeof part.name === 'string' && part.arguments !== undefined,
        'INVALID_MODEL_RESULT', 'Tool calls require id, name and arguments');
    }
  }
  return message;
}

/** One long-lived official Pi Agent per group. Python owns provider I/O and durability. */
class Session {
  constructor(params, callback) {
    check(typeof params.system === 'string', 'INVALID_SYSTEM', 'init.system must be a string');
    check(Number.isSafeInteger(params.revision ?? 0) && (params.revision ?? 0) >= 0,
      'INVALID_REVISION', 'revision must be a non-negative integer');
    this.id = params.sessionId;
    this.system = params.system;
    this.callback = callback;
    this.revision = params.revision ?? 0;
    this.pendingPersistence = new Set();
    this.transition = false;
    this.toolTransition = false;
    this.poisoned = undefined;
    this.closing = false;
    this.closePromise = undefined;
    this.runPromise = undefined;
    this.terminateRequested = false;
    this.maintenanceRequired = params.maintenanceRequired === true;
    this.continuationRequired = this.maintenanceRequired && params.continuationRequired === true;
    this.tools = new Map();
    const messages = validateHistory(params.messages ?? [], this.system);
    const declarations = params.tools ?? getCurrentTools(messages);
    check(Array.isArray(declarations), 'INVALID_TOOL', 'init.tools must be an array');
    for (const tool of declarations) this.addDefinition(tool);
    for (const declared of getCurrentTools(messages)) {
      check(canonical(this.tools.get(declared.name)) === canonical(normalizeTool(declared)),
        'TOOL_IMMUTABLE', `init.tools removes or changes replayed tool ${declared.name}`);
    }
    this.agent = new Agent({
      initialState: {
        systemPrompt: this.system,
        model: HOST_MODEL,
        messages,
        tools: [...this.tools.values()].map((tool) => this.executableTool(tool)),
      },
      sessionId: this.id,
      toolExecution: 'sequential', steeringMode: 'all', followUpMode: 'all',
      // Identity conversion preserves every custom native field; no history pruning or rewriting.
      convertToLlm: (history) => history,
      streamFn: (model, context, options) => this.modelStream(model, context, options),
      beforeToolCall: async () => this.terminateRequested
        ? { block: true, reason: 'Run terminated by an earlier tool', terminate: true }
        : undefined,
      finishTurn: async ({ message, toolResults }, signal) => {
        // Provider-reported pressure requests a safe maintenance boundary, not an
        // abort: every tool in the current assistant batch has already finalized.
        if (message.hostMaintenanceRequired === true) {
          this.maintenanceRequired = true;
          this.continuationRequired = !this.terminateRequested && !signal?.aborted
            && (toolResults.length > 0 || (message.hostControl === 'capacity_admission' && message.hostContinuationRequired === true));
        }
        return this.terminateRequested || this.maintenanceRequired || signal?.aborted
          ? { action: 'end' }
          : undefined;
      },
      // Pi snapshots executable tools at run entry. Update only the next-turn tool set;
      // its own declareToolChanges appends the corresponding transcript declaration.
      prepareNextTurnWithContext: async ({ context }) => ({
        context: { ...context, tools: this.agent.state.tools.slice() },
      }),
    });
    this.agent.subscribe(async (event) => {
      await this.persist(event);
    });
  }

  addDefinition(raw) {
    const tool = normalizeTool(raw);
    const existing = this.tools.get(tool.name);
    check(!existing || canonical(existing) === canonical(tool),
      'TOOL_IMMUTABLE', `Tool ${tool.name} cannot be redefined; append a new name`);
    if (!existing) this.tools.set(tool.name, tool);
    return !existing;
  }

  executableTool(tool) {
    return {
      ...clone(tool), label: tool.name, executionMode: 'sequential',
      execute: async (toolCallId, args, signal) => {
        const result = await this.callback('tool', {
          sessionId: this.id, toolCallId, name: tool.name, arguments: clone(args),
        }, signal);
        check(result && Array.isArray(result.content), 'INVALID_TOOL_RESULT', 'Tool callback must return a content array');
        if (result.terminate === true) this.terminateRequested = true;
        return clone(result);
      },
    };
  }

  modelStream(model, context, options) {
    const stream = createAssistantMessageEventStream();
    // Only adapting one response to Pi's event stream; Pi itself owns the reasoning/tool loop.
    void (async () => {
      try {
        if (options?.signal?.aborted) throw new DOMException('Operation aborted', 'AbortError');
        const result = await this.callback('model', {
          sessionId: this.id, system: this.system,
          messages: clone(context.messages), tools: clone(getCurrentTools(context.messages)),
        }, options?.signal);
        const message = normalizeAssistant(result, model);
        if (message.stopReason === 'error' || message.stopReason === 'aborted') {
          stream.push({ type: 'error', reason: message.stopReason, error: message });
        } else {
          stream.push({ type: 'done', reason: message.stopReason, message });
        }
      } catch (error) {
        const aborted = options?.signal?.aborted || error?.name === 'AbortError';
        const message = {
          role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id,
          usage: clone(EMPTY_USAGE), stopReason: aborted ? 'aborted' : 'error',
          hostSynthetic: 'provider_callback_failure', hostControl: 'provider_failure', native: { items: [] },
          errorMessage: error instanceof Error ? error.message : String(error), timestamp: Date.now(),
        };
        stream.push({ type: 'error', reason: message.stopReason, error: message });
      }
    })();
    return stream;
  }

  persist(event) {
    this.assertHealthy();
    const snapshot = clone(event);
    const revision = ++this.revision;
    // Reserve ordering before invoking the host, but do not serialize acknowledgments:
    // an awaited message_end handler may itself append a newly durable group message.
    // Serializing the nested input_queued callback behind its parent deadlocks the RPC.
    const operation = (async () => {
      try {
        // No run AbortSignal: canceled runs still require durable terminal events.
        await this.callback('event', { sessionId: this.id, revision, event: snapshot });
      } catch (error) {
        this.poisoned = `Persistence callback failed: ${error instanceof Error ? error.message : String(error)}`;
        this.agent.abort();
        throw new RuntimeError('PERSISTENCE_FAILED', this.poisoned);
      }
    })();
    this.pendingPersistence.add(operation);
    void operation.then(
      () => this.pendingPersistence.delete(operation),
      () => this.pendingPersistence.delete(operation),
    );
    return operation;
  }

  unresolvedCalls() {
    const pending = new Map();
    for (const message of this.agent.state.messages) {
      if (message.role === 'assistant') {
        for (const part of message.content) {
          if (part.type === 'toolCall') pending.set(part.id, part);
        }
      } else if (message.role === 'toolResult') {
        pending.delete(message.toolCallId);
      }
    }
    return [...pending.values()];
  }

  assertHealthy() {
    check(!this.poisoned, 'PERSISTENCE_FAILED', this.poisoned);
  }

  assertWritable() {
    this.assertHealthy();
    check(!this.closing, 'SESSION_CLOSING', 'Session is closing');
    check(!this.transition, 'SESSION_TRANSITION', 'Session context transition is in progress');
  }

  snapshot() {
    return clone({
      sessionId: this.id, system: this.system, revision: this.revision,
      tools: [...this.tools.values()], messages: this.agent.state.messages,
      queuedMessages: this.agent.peekQueuedMessages(),
      isRunning: this.agent.state.isStreaming,
      pendingToolCalls: [...this.agent.state.pendingToolCalls],
      paused: !this.agent.state.isStreaming && this.unresolvedCalls().length > 0,
      maintenanceRequired: this.maintenanceRequired,
      continuationRequired: this.continuationRequired,
      terminatedByTool: this.terminateRequested,
      unresolvedToolCalls: this.unresolvedCalls(),
      error: this.agent.state.errorMessage ?? null, poisoned: this.poisoned ?? null,
    });
  }

  async enqueue(additions) {
    this.assertWritable();
    const messages = normalizeAdditions(additions);
    await this.persist({ type: 'input_queued', messages });
    // No await between final health check and queue mutation. Arrivals during final event
    // persistence remain queued for the next explicit continuation rather than disappearing.
    this.assertHealthy();
    for (const message of messages) this.agent.steer(message);
    return { ...this.snapshot(), queued: true };
  }

  async run(method, additions) {
    this.assertWritable();
    if (method === 'prompt' && this.agent.state.isStreaming) return this.enqueue(additions);
    check(!this.agent.state.isStreaming, 'SESSION_BUSY', 'Session is already running');
    check(this.unresolvedCalls().length === 0, 'UNRESOLVED_TOOL_CALLS',
      'Session is paused: host must resolve interrupted tool calls before another model request');
    check(!this.maintenanceRequired, 'MAINTENANCE_REQUIRED',
      'Explicit host compaction is required before another model request');
    let messages = method === 'prompt' ? normalizeAdditions(additions) : undefined;
    if (method === 'prompt' && this.agent.hasQueuedMessages()) {
      // A terminating tool can leave earlier arrivals queued. Official prompt()
      // appends its argument before polling steering, so explicitly preserve FIFO.
      messages = [...this.agent.peekQueuedMessages(), ...messages];
      this.agent.clearSteeringQueue();
    }
    this.terminateRequested = false;
    const last = this.agent.state.messages.at(-1);
    const hostAdmissionBoundary = last?.hostControl === 'capacity_admission'
      && last.hostContinuationRequired === true;
    const execution = method === 'prompt'
      ? this.agent.prompt(messages)
      : this.agent.hasQueuedMessages() || hostAdmissionBoundary ? this.agent.prompt([]) : this.agent.continue();
    this.runPromise = execution;
    try {
      await execution;
      this.assertHealthy();
      return this.snapshot();
    } finally {
      if (this.runPromise === execution) this.runPromise = undefined;
    }
  }

  async addTools(rawTools) {
    this.assertWritable();
    check(Array.isArray(rawTools), 'INVALID_TOOL', 'tools must be an array');
    const additions = rawTools.map(normalizeTool);
    const candidate = new Map(this.tools);
    for (const tool of additions) {
      const existing = candidate.get(tool.name);
      check(!existing || canonical(existing) === canonical(tool),
        'TOOL_IMMUTABLE', `Tool ${tool.name} cannot be redefined; append a new name`);
      candidate.set(tool.name, tool);
    }
    const novel = [...candidate.values()].filter((tool) => !this.tools.has(tool.name));
    if (novel.length > 0) {
      check(!this.toolTransition, 'SESSION_TRANSITION', 'A tool declaration update is in progress');
      this.toolTransition = true;
      try {
        await this.persist({ type: 'tools_added', tools: novel });
        for (const tool of novel) this.addDefinition(tool);
        this.agent.state.tools = [...this.tools.values()].map((tool) => this.executableTool(tool));
      } finally {
        this.toolTransition = false;
      }
    }
    return this.snapshot();
  }

  async replaceContext(params) {
    this.assertWritable();
    check(!this.agent.state.isStreaming && !this.agent.hasQueuedMessages(),
      'SESSION_BUSY', 'Compaction requires an idle session with no queued messages');
    check(Number.isSafeInteger(params.expectedRevision) && params.expectedRevision === this.revision,
      'REVISION_CONFLICT', 'Compaction revision does not match current session');
    check(typeof params.reason === 'string' && params.reason.length > 0,
      'INVALID_COMPACTION', 'Explicit compaction requires a reason');
    const messages = validateHistory(params.messages, this.system);
    check(messages[0]?.role === 'system', 'INVALID_COMPACTION', 'Compaction must preserve a leading system baseline');
    const sortTools = (tools) => tools.slice().sort((left, right) => left.name.localeCompare(right.name));
    check(canonical(sortTools(getCurrentTools(messages))) === canonical(sortTools([...this.tools.values()])),
      'TOOL_IMMUTABLE', 'Compaction must preserve all current tool declarations');
    check(this.pendingPersistence.size === 0 && !this.toolTransition,
      'SESSION_TRANSITION', 'Compaction cannot overlap pending event or tool transactions');
    const expectedRevision = this.revision;
    this.transition = true;
    try {
      await this.persist({ type: 'context_replaced', expectedRevision, reason: params.reason, messages });
      // The reservation prevents a successful host ACK and a failed local swap from diverging.
      // Old events remain in the durable ledger; only the explicit active view is replaced.
      this.agent.state.messages = messages;
      this.maintenanceRequired = false;
      this.continuationRequired = false;
      return this.snapshot();
    } finally {
      this.transition = false;
    }
  }

  async resolveTools(params) {
    this.assertWritable();
    check(!this.agent.state.isStreaming && this.pendingPersistence.size === 0,
      'SESSION_BUSY', 'Tool recovery requires an idle, settled session');
    check(params.expectedRevision === this.revision, 'REVISION_CONFLICT', 'Recovery revision does not match');
    check(Array.isArray(params.results) && params.results.length > 0,
      'INVALID_RECOVERY', 'results must contain host-resolved toolResult messages');
    const outstanding = new Map(this.unresolvedCalls().map((call) => [call.id, call]));
    const results = clone(params.results);
    for (const result of results) {
      const call = outstanding.get(result.toolCallId);
      check(result.role === 'toolResult' && call && result.toolName === call.name,
        'INVALID_RECOVERY', 'Each recovery result must match one outstanding tool call');
      check(Array.isArray(result.content) && typeof result.isError === 'boolean',
        'INVALID_RECOVERY', 'Recovery results require content and isError');
      result.timestamp ??= Date.now();
      outstanding.delete(result.toolCallId);
    }
    this.transition = true;
    try {
      for (const result of results) {
        await this.persist({ type: 'message_start', message: result, recovery: true });
        await this.persist({ type: 'message_end', message: result, recovery: true });
        this.agent.state.messages.push(result);
      }
      return this.snapshot();
    } finally {
      this.transition = false;
    }
  }

  async abort() {
    const wasRunning = this.agent.state.isStreaming;
    this.agent.abort();
    await this.agent.waitForIdle();
    await Promise.all([...this.pendingPersistence]);
    this.assertHealthy();
    if (wasRunning) await this.persist({ type: 'run_aborted', unresolvedToolCalls: this.unresolvedCalls() });
    return this.snapshot();
  }

  close() {
    if (this.closePromise) return this.closePromise;
    this.closing = true;
    this.closePromise = (async () => {
      await this.abort();
      await this.persist({ type: 'session_closed' });
      return this.snapshot();
    })();
    return this.closePromise;
  }

}

export class PiRuntime {
  constructor(callback) {
    check(typeof callback === 'function', 'INVALID_CALLBACK', 'A host callback is required');
    this.callback = callback;
    this.sessions = new Map();
  }

  async handle(method, params = {}) {
    check(typeof params.sessionId === 'string' && params.sessionId.length > 0,
      'INVALID_SESSION', 'sessionId is required');
    if (method === 'init') {
      check(!this.sessions.has(params.sessionId), 'SESSION_EXISTS', 'Session already initialized; use its existing Agent');
      const session = new Session(params, this.callback);
      this.sessions.set(params.sessionId, session);
      return session.snapshot();
    }
    const session = this.sessions.get(params.sessionId);
    if (!session && method === 'close') return { sessionId: params.sessionId, closed: true, alreadyClosed: true };
    check(session, 'SESSION_NOT_FOUND', 'Session is not initialized');
    switch (method) {
      case 'state': return session.snapshot();
      case 'prompt': return session.run('prompt', params.additions);
      case 'append': return session.enqueue(params.additions);
      case 'continue': return session.run('continue');
      case 'add_tools': return session.addTools(params.tools);
      case 'replace_context': return session.replaceContext(params);
      case 'resolve_tools': return session.resolveTools(params);
      case 'abort': return session.abort();
      case 'close': {
        const state = await session.close();
        this.sessions.delete(params.sessionId);
        return { ...state, closed: true };
      }
      default: throw new RuntimeError('UNKNOWN_METHOD', `Unknown method: ${method}`);
    }
  }

  /** Transport shutdown cannot persist through a dead parent. Never restart or replay tools. */
  shutdown() {
    for (const session of this.sessions.values()) session.agent.abort();
  }
}
