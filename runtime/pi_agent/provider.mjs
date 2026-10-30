/** MaiBot config -> official Pi providers. No Python inference callback or disk credentials. */
import { createHash } from 'node:crypto';
import { AsyncLocalStorage } from 'node:async_hooks';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { streamSimple } from '@earendil-works/pi-ai/compat';
import { openAIDecisionsApi } from '@earendil-works/pi-ai/api/openai-decisions.lazy';
import { makeImageProvider } from './image-provider.mjs';

function memoryCredentials() {
  const values = new Map();
  return {
    read: async (id) => values.get(id),
    list: async () => [...values.entries()].map(([providerId, credential]) => ({ providerId, type: credential.type })),
    modify: async (id, fn) => { const value = await fn(values.get(id)); if (value == null) values.delete(id); else values.set(id, value); return value; },
    delete: async (id) => { values.delete(id); },
  };
}
const states = new WeakMap();
const API = { openai: 'openai-completions', openai_responses: 'openai-responses', gemini: 'google-generative-ai', anthropic: 'anthropic-messages' };
const LEVELS = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
const positive = (value) => Number.isSafeInteger(value) && value > 0;
const canonical = (value) => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
const fingerprint = (value) => createHash('sha256').update(canonical(value)).digest('hex');
const error = (code, message) => Object.assign(new Error(message), { code });

// Pi 1.4's Google adapter only accepts global fetch; @google/genai has no injectable
// transport. AsyncLocalStorage scopes this compatibility bridge to its own request,
// retaining official Pi conversion/replay and never mixing simultaneous credentials.
const googleTransport = new AsyncLocalStorage();
let unscopedFetch;
function installGoogleTransport() {
  if (unscopedFetch) return;
  unscopedFetch = globalThis.fetch;
  globalThis.fetch = function scopedGoogleFetch(input, init) {
    const scoped = googleTransport.getStore();
    return (scoped?.matches(input) ? scoped.fetch : unscopedFetch)(input, init);
  };
}

export function splitExtraParams(extra = {}) {
  const { headers = {}, query = {}, body = {}, pi = {}, ...flat } = extra;
  return { headers, query, pi, body: { ...body, ...flat } };
}

/** Resolve host auth/query/header conventions in the Node transport, including mock fetch. */
export function createProviderFetch(provider, extra = {}, fetchImpl) {
  const transport = fetchImpl ?? unscopedFetch ?? globalThis.fetch;
  return async (input, init = {}) => {
    const request = input instanceof Request ? input : undefined;
    const url = new URL(request ? request.url : String(input));
    for (const [key, value] of Object.entries({ ...provider.default_query, ...extra.query })) {
      if (value != null) url.searchParams.set(key, String(value));
    }
    const headers = new Headers(request?.headers);
    for (const [key, value] of new Headers(init.headers)) headers.set(key, value);
    for (const [key, value] of Object.entries({ ...provider.default_headers, ...extra.headers })) headers.set(key, String(value));
    const auth = provider.auth_type ?? 'bearer';
    if (auth !== 'bearer') {
      headers.delete('authorization'); headers.delete('x-api-key'); headers.delete('x-goog-api-key');
      url.searchParams.delete('key');
    }
    if (auth === 'header' && provider.api_key) {
      headers.set(provider.auth_header_name || 'Authorization', [provider.auth_header_prefix, provider.api_key].filter(Boolean).join(' '));
    } else if (auth === 'query' && provider.api_key) url.searchParams.set(provider.auth_query_name || 'api_key', provider.api_key);
    else if (!['bearer', 'none', 'header', 'query'].includes(auth)) throw error('UNSUPPORTED_AUTH', `Unsupported auth type: ${auth}`);
    if (provider.organization) headers.set('OpenAI-Organization', provider.organization);
    if (provider.project) headers.set('OpenAI-Project', provider.project);
    const base = request ? new Request(url, request) : url;
    (init.signal ?? request?.signal)?.throwIfAborted();
    // Custom API-key headers are not stripped by fetch on cross-origin redirects.
    // Never let an authenticated provider request follow an implicit redirect.
    return transport(base, { ...init, headers, redirect: 'error' });
  };
}

function validateExtraBody(body) {
  for (const key of ['model', 'input', 'messages', 'contents', 'tools', 'system', 'instructions', 'stream', 'httpOptions', 'abortSignal']) {
    if (key in body) throw error('RESERVED_MODEL_PARAMETER', `extra_params cannot override request identity field: ${key}`);
  }
  if (body.fallbacks?.length) throw error('UNSUPPORTED_MODEL_FALLBACK', 'Automatic model fallback is not permitted');
}
function configuredOutput(body) {
  return body.max_output_tokens ?? body.max_completion_tokens ?? body.max_tokens ?? body.config?.maxOutputTokens ?? body.generationConfig?.maxOutputTokens;
}
function validateGoogleConfig(config = {}) {
  for (const key of ['httpOptions', 'http_options', 'abortSignal', 'abort_signal', 'model', 'contents', 'tools', 'systemInstruction']) {
    if (key in config && key !== 'abortSignal' && key !== 'tools' && key !== 'systemInstruction') throw error('RESERVED_MODEL_PARAMETER', `Google config cannot override transport/identity field: ${key}`);
  }
}
function mergeProviderPayload(payload, rawBody, api) {
  const { reasoning_effort, max_tokens, max_completion_tokens, max_output_tokens, temperature, tool_choice, ...body } = rawBody;
  if (api === 'google-generative-ai') {
    const { config = {}, generationConfig = {}, thinking_budget, include_thoughts, reasoning, ...flat } = body;
    const converted = Object.fromEntries(Object.entries(flat).map(([key, value]) => [key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()), value]));
    const custom = { ...converted, ...generationConfig, ...config };
    for (const key of ['httpOptions', 'http_options', 'abortSignal', 'abort_signal', 'model', 'contents', 'tools', 'systemInstruction']) {
      if (key in custom) throw error('RESERVED_MODEL_PARAMETER', `Google extras cannot override transport/identity field: ${key}`);
    }
    const merged = { ...payload, config: { ...payload.config, ...custom } };
    if (thinking_budget != null || include_thoughts != null) merged.config.thinkingConfig = { ...merged.config.thinkingConfig,
      ...(thinking_budget == null ? {} : { thinkingBudget: thinking_budget }), ...(include_thoughts == null ? {} : { includeThoughts: include_thoughts }) };
    if (temperature != null) merged.config.temperature = temperature;
    return merged;
  }
  const merged = { ...payload, ...body };
  if (body.reasoning && payload.reasoning) merged.reasoning = { ...payload.reasoning, ...body.reasoning };
  if (temperature != null) merged.temperature = temperature;
  return merged;
}

/** No number inferred from a marketing name or a similar public model. */
function capabilityFor(info, extra) {
  const admission = info.request_admission ?? {};
  const contextWindow = admission.context_window ?? extra.pi.contextWindow ?? null;
  const outputLimit = admission.max_output_tokens ?? extra.pi.maxTokens ?? null;
  return { ...admission, mode: admission.mode ?? 'audit', contextWindow, outputLimit,
    contextKnown: positive(contextWindow), source: admission.capability_source || extra.pi.capabilitySource || null };
}

const WIRE_FIELDS = {
  'openai-responses': new Set('model input instructions tools tool_choice parallel_tool_calls max_output_tokens reasoning temperature top_p text store stream include service_tier prompt_cache_key prompt_cache_retention prompt_cache_options metadata safety_identifier user truncation background'.split(' ')),
  'openai-completions': new Set('model messages tools tool_choice parallel_tool_calls max_tokens max_completion_tokens reasoning reasoning_effort temperature top_p top_k min_p frequency_penalty presence_penalty repetition_penalty stop n seed logit_bias logprobs top_logprobs response_format stream stream_options store service_tier metadata user prompt_cache_key prompt_cache_retention'.split(' ')),
  'anthropic-messages': new Set('model messages system tools tool_choice max_tokens thinking output_config temperature top_p top_k stop_sequences stream metadata betas'.split(' ')),
  'google-generative-ai': new Set('model contents config generationConfig systemInstruction tools toolConfig safetySettings'.split(' ')),
  'google-audio-transcription': new Set('contents generationConfig'.split(' ')),
};
const GOOGLE_CONFIG_FIELDS = new Set('abortSignal temperature topP topK candidateCount maxOutputTokens stopSequences presencePenalty frequencyPenalty seed responseMimeType responseSchema responseJsonSchema responseLogprobs logprobs thinkingConfig systemInstruction tools toolConfig safetySettings responseModalities mediaResolution'.split(' '));
/** Structural visibility only, not a tokenizer. Schema property names are plain
 * input text; actual media, opaque state and server-side expansion are not. */
function inputVisibilityKnown(payload, api) {
  const fields = WIRE_FIELDS[api];
  if (!fields || Object.keys(payload).some((key) => !fields.has(key))) return false;
  const textBlocks = (content, protocol) => {
    if (content == null || typeof content === 'string') return true;
    if (!Array.isArray(content)) return false;
    return content.every((part) => {
      if (!part || typeof part !== 'object') return false;
      if (protocol === 'google') {
        if (part.thoughtSignature || part.inlineData || part.fileData) return false;
        return typeof part.text === 'string' || !!part.functionCall || (!!part.functionResponse && !part.functionResponse.parts);
      }
      if (['text', 'input_text', 'output_text'].includes(part.type)) return typeof part.text === 'string';
      if (protocol === 'anthropic' && part.type === 'tool_use') return true;
      if (protocol === 'anthropic' && part.type === 'tool_result') return textBlocks(part.content, protocol);
      return false;
    });
  };
  const toolsKnown = (tools) => tools == null || (Array.isArray(tools) && tools.every((tool) => {
    if (api.startsWith('google')) return Array.isArray(tool.functionDeclarations);
    return tool.type === 'function' || (api === 'anthropic-messages' && tool.type == null && typeof tool.name === 'string');
  }));
  if (api.startsWith('google')) {
    const config = payload.config ?? payload.generationConfig ?? {};
    if (Object.keys(config).some((key) => !GOOGLE_CONFIG_FIELDS.has(key))) return false;
    if (!toolsKnown(payload.tools ?? config.tools)) return false;
    const system = payload.systemInstruction ?? config.systemInstruction;
    if (system && typeof system !== 'string' && !textBlocks(system.parts, 'google')) return false;
    return (payload.contents ?? []).every((entry) => textBlocks(entry.parts, 'google'));
  }
  if (!toolsKnown(payload.tools)) return false;
  if (api === 'openai-responses') {
    if (typeof payload.input === 'string' || payload.input == null) return true;
    return Array.isArray(payload.input) && payload.input.every((item) => {
      if (item.type === 'function_call') return true;
      if (item.type === 'function_call_output') return textBlocks(item.output, 'openai');
      return (!item.type || item.type === 'message') && textBlocks(item.content, 'openai');
    });
  }
  const protocol = api === 'anthropic-messages' ? 'anthropic' : 'openai';
  return textBlocks(payload.system, protocol) && (payload.messages ?? []).every((message) =>
    !message.audio && !message.reasoning_details && textBlocks(message.content, protocol));
}

/** Admission is honest about unknown input/reasoning accounting; audit never promises safety. */
export function checkRequestAdmission(payload, capability, api, wireBytes) {
  const output = payload.max_output_tokens ?? payload.max_completion_tokens ?? payload.max_tokens ?? payload.config?.maxOutputTokens ?? payload.generationConfig?.maxOutputTokens;
  if (positive(capability.outputLimit) && positive(output) && output > capability.outputLimit) throw error('CONTEXT_CAPACITY', 'Configured output exceeds the declared model output limit');
  if (capability.contextKnown && positive(output) && output + (capability.safety_margin_tokens ?? 256) > capability.contextWindow) throw error('CONTEXT_CAPACITY', 'Output allowance alone exceeds the declared context window');
  const strict = capability.mode === 'strict';
  const serialized = wireBytes ?? JSON.stringify(payload);
  const visible = inputVisibilityKnown(payload, api);
  const hasBound = capability.input_counter === 'utf8_bytes_bound' && positive(capability.tokens_per_utf8_byte_upper)
    && Number.isSafeInteger(capability.protocol_overhead_tokens) && capability.protocol_overhead_tokens >= 0 && visible && !capability.transportUnverified;
  const boundedInput = hasBound ? Buffer.byteLength(serialized) * capability.tokens_per_utf8_byte_upper + capability.protocol_overhead_tokens : null;
  const outputKnown = positive(output) && capability.output_semantics === 'includes_reasoning';
  const margin = capability.safety_margin_tokens ?? 256;
  if (boundedInput != null && capability.contextKnown && boundedInput + margin > capability.contextWindow) throw error('CONTEXT_CAPACITY', 'Declared input bound alone exceeds the context window');
  if (boundedInput != null && positive(capability.max_input_tokens) && boundedInput + margin > capability.max_input_tokens) throw error('CONTEXT_CAPACITY', 'Declared input bound exceeds the input window');
  if (boundedInput != null && outputKnown && capability.contextKnown && boundedInput + output + margin > capability.contextWindow) throw error('CONTEXT_CAPACITY', 'Declared input/output bound exceeds the context window');
  if (strict && (!capability.source || boundedInput == null || !outputKnown || (!capability.contextKnown && !(positive(capability.max_input_tokens) && positive(capability.outputLimit))))) {
    throw error('CONTEXT_CAPABILITY_UNKNOWN', `Strict capacity admission cannot establish a safe ${api} request bound`);
  }
  return { mode: capability.mode, boundedInput, output: output ?? null, verified: hasBound && outputKnown && capability.contextKnown };
}

/** Build one process-local Pi model runtime for every host task; secrets never touch sessions. */
export async function createHostModelRuntime(config, { fetch: fetchImpl } = {}) {
  if (!config || !Array.isArray(config.models) || !Array.isArray(config.api_providers)) throw error('INVALID_ROUTE_CONFIG', 'models and api_providers are required');
  config = structuredClone(config);
  const runtime = await ModelRuntime.create({ credentials: memoryCredentials(), modelsPath: null, allowModelNetwork: false, refreshOnCreate: false });
  const providers = new Map(config.api_providers.map((provider) => [provider.name, provider]));
  const routes = new Map();
  const state = { config, providers, routes, counts: new Map(), fetchImpl };
  states.set(runtime, state);
  for (const info of config.models) {
    const provider = providers.get(info.api_provider);
    if (!provider) throw error('INVALID_ROUTE_CONFIG', `Missing provider for model ${info.name}`);
    if (routes.has(info.name)) throw error('INVALID_ROUTE_CONFIG', `Duplicate host model name: ${info.name}`);
    const extra = splitExtraParams(info.extra_params);
    const capability = { ...capabilityFor(info, extra), transportUnverified:
      Object.keys({ ...provider.default_headers, ...extra.headers }).length > 0
      || Object.keys({ ...provider.default_query, ...extra.query }).length > 0 };
    // Named/exact transport counters from the retired Python path are not
    // implemented here. A matching hash alone cannot make byte bounds exact.
    if (extra.pi.compat?.allowedFallbackModels?.length || extra.body.fallbacks?.length) throw error('UNSUPPORTED_MODEL_FALLBACK', 'Automatic model fallback is not permitted');
    validateExtraBody(extra.body);
    const api = extra.pi.api ?? API[provider.client_type];
    const route = { info, provider, extra, capability, api };
    routes.set(info.name, route);
    // Pi 1.4 has a native classifier operation. Register it separately so the
    // same Luna alias can keep serving chat/memory without changing their API.
    if (info.model_identifier === 'gpt-6-luna' && (info.model_type ?? 'llm') === 'llm'
        && ['openai', 'openai_responses'].includes(provider.client_type) && !extra.pi.operation) {
      const classifierId = `maibot:${encodeURIComponent(provider.name)}:${encodeURIComponent(info.name)}:decisions`;
      route.classifierProviderId = classifierId;
      runtime.registerProvider(classifierId, {
        baseUrl: provider.base_url, authHeader: (provider.auth_type ?? 'bearer') === 'bearer',
        models: [{ type: 'classifier', id: info.model_identifier, name: info.name, api: 'openai-decisions',
          input: ['text'], contextWindow: capability.contextKnown ? capability.contextWindow : 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
        classifiers: { 'openai-decisions': openAIDecisionsApi() },
      });
      await runtime.setRuntimeApiKey(classifierId, provider.api_key || 'maibot-no-auth');
    }
    if (extra.pi.operation === 'image') {
      const imageApi = extra.pi.api ?? (provider.client_type === 'gemini' ? 'maibot-google-images' : ['openai', 'openai_responses'].includes(provider.client_type) ? 'maibot-openai-images' : undefined);
      if (!imageApi) throw error('UNSUPPORTED_IMAGE_PROVIDER', `Image API must be configured for ${info.name}`);
      const imageProviderId = `maibot:${encodeURIComponent(provider.name)}:${encodeURIComponent(info.name)}`;
      route.providerId = imageProviderId; route.api = imageApi; route.operation = 'image';
      runtime.registerProvider(imageProviderId, { baseUrl: provider.base_url, authHeader: (provider.auth_type ?? 'bearer') === 'bearer',
        models: [{ type: 'image', id: info.model_identifier, name: info.name, api: imageApi, input: ['text', 'image'], output: ['text', 'image'], cost: { input: info.price_in ?? 0, output: info.price_out ?? 0, cacheRead: 0, cacheWrite: 0 } }],
        images: { [imageApi]: makeImageProvider(provider, extra, createProviderFetch(provider, extra, fetchImpl), fetchImpl, capability) },
      });
      await runtime.setRuntimeApiKey(imageProviderId, provider.api_key || 'maibot-no-auth');
      continue;
    }
    if ((info.model_type ?? 'llm') !== 'llm') continue; // Auxiliary operations are explicitly non-chat adapters.
    if (!api) continue; // Unknown plugin protocols fail only when selected, never silently map to OpenAI.
    const id = `maibot:${encodeURIComponent(provider.name)}:${encodeURIComponent(info.name)}`;
    route.providerId = id;
    const effort = extra.body.reasoning_effort ?? extra.body.reasoning?.effort ?? extra.pi.thinkingLevel;
    if (effort != null && !LEVELS.includes(effort)) throw error('UNSUPPORTED_REASONING', `Unsupported configured effort for ${info.name}`);
    const thinkingLevelMap = extra.pi.thinkingLevelMap ?? (effort ? { [effort]: effort } : undefined);
    const taskBudgets = Object.values(config.model_task_config ?? {}).filter((task) => task.model_list?.includes(info.name))
      .map((task) => task.max_tokens).filter((value) => value != null);
    if (taskBudgets.some((value) => !positive(value))) throw error('INVALID_OUTPUT_LIMIT', 'Configured task output budgets must be positive integers');
    // Zero is registration metadata only; resolveTaskModel refuses to infer a
    // budget when neither the model, task nor caller configured one.
    const maxTokens = info.max_tokens ?? configuredOutput(extra.body) ?? capability.outputLimit
      ?? (taskBudgets.length ? Math.max(...taskBudgets) : 0);
    if (maxTokens !== 0 && !positive(maxTokens)) throw error('INVALID_OUTPUT_LIMIT', 'Configured model output budget must be a positive integer');
    runtime.registerProvider(id, {
      baseUrl: provider.base_url, api, authHeader: (provider.auth_type ?? 'bearer') === 'bearer',
      models: [{ id: info.model_identifier, name: info.name, api, reasoning: effort != null && effort !== 'off', thinkingLevelMap,
        input: info.visual ? ['text', 'image'] : ['text'],
        // Pi explicitly supports 0 for unknown/unbounded estimator metadata. This
        // is not a claimed capacity; strict final admission still refuses unknowns.
        contextWindow: capability.contextKnown ? capability.contextWindow : 0,
        maxTokens, cost: { input: info.price_in ?? 0, output: info.price_out ?? 0, cacheRead: info.cache_price_in ?? 0, cacheWrite: info.price_in ?? 0 },
        ...(extra.pi.compat ? { compat: extra.pi.compat } : {}),
      }],
      streamSimple: (model, context, options = {}) => {
        for (const message of context.messages) for (const tool of message.toolsAdded ?? []) {
          if (tool.constrainedSampling?.strict === 'prefer') throw error('UNSUPPORTED_SCHEMA_FALLBACK', 'Use required strict schemas or explicit non-strict schemas; automatic schema fallback is not permitted');
        }
        const taskOptions = model.hostRequestOptions ?? {};
        const dispatch = createProviderFetch(provider, extra, fetchImpl);
        const transport = async (input, init = {}) => {
          // Google onPayload receives SDK params, not its final REST body. Run
          // capacity admission again on the actual serialized HTTP payload so
          // SDK field translation is inside the checked boundary as well.
          const serialized = typeof init.body === 'string' ? init.body : input instanceof Request ? await input.clone().text() : null;
          if (serialized == null) throw error('UNSUPPORTED_WIRE_PAYLOAD', 'Cannot validate the provider HTTP payload');
          checkRequestAdmission(JSON.parse(serialized), capability, api, serialized);
          return dispatch(input, init);
        };
        const timeoutMs = options.timeoutMs ?? (provider.timeout ? provider.timeout * 1000 : undefined);
        const effective = { ...options, ...taskOptions,
          // Automatic moving cache breakpoints rewrite old Anthropic blocks.
          // Stable native histories take precedence over implicit prompt caching.
          cacheRetention: 'none', maxRetries: 0,
          signal: Number.isFinite(timeoutMs) && timeoutMs > 0
            ? AbortSignal.any([...(options.signal ? [options.signal] : []), AbortSignal.timeout(Math.ceil(timeoutMs))])
            : options.signal };
        const outputBudget = taskOptions.maxTokens ?? options.maxTokens ?? maxTokens;
        const hostHook = options.onPayload;
        effective.onPayload = async (payload, activeModel) => {
          let merged = mergeProviderPayload(payload, model.hostExtraBody ?? extra.body, api);
          if (hostHook) merged = (await hostHook(merged, activeModel)) ?? merged;
          // Some gateways return DeepSeek thinking as `reasoning`. Pi preserves
          // that native signature, but DeepSeek continuation reads reasoning_content.
          // Adapt only the outgoing protocol; keep the SDK transcript untouched.
          if (api === 'openai-completions' && extra.pi.compat?.thinkingFormat === 'deepseek') {
            merged.messages = merged.messages.map((message) => {
              if (message.role !== 'assistant' || typeof message.reasoning !== 'string' || message.reasoning_content) return message;
              const { reasoning, ...rest } = message;
              return { ...rest, reasoning_content: reasoning };
            });
          }
          if (merged.model !== model.id) throw error('RESERVED_MODEL_PARAMETER', 'Payload hook cannot replace the selected model');
          // Restore after both SDK estimation and user hooks. Alternate budget keys
          // cannot bypass admission or silently reduce the configured allowance.
          for (const key of ['max_tokens', 'max_completion_tokens', 'max_output_tokens', 'maxOutputTokens']) delete merged[key];
          if (api === 'google-generative-ai') {
            merged.config = { ...merged.config, maxOutputTokens: outputBudget };
            delete merged.generationConfig;
            validateGoogleConfig(merged.config);
          } else if (api === 'openai-responses') merged.max_output_tokens = outputBudget;
          else if (api === 'openai-completions') merged['max_completion_tokens' in payload ? 'max_completion_tokens' : 'max_tokens'] = outputBudget;
          else if (api === 'anthropic-messages') merged.max_tokens = outputBudget;
          const sampling = api === 'google-generative-ai' ? merged.config : merged;
          if (info.send_temperature === false) delete sampling.temperature;
          else if (effective.temperature != null) sampling.temperature = effective.temperature;
          const reasoning = effective.reasoning ?? effort;
          if (merged.fallbacks?.length) throw error('UNSUPPORTED_MODEL_FALLBACK', 'Payload cannot enable automatic model fallback');
          if (api === 'openai-responses' && reasoning && reasoning !== 'off') merged.reasoning = { ...merged.reasoning, effort: reasoning };
          else if (api === 'openai-completions' && reasoning && reasoning !== 'off') merged.reasoning_effort = reasoning;
          if (reasoning === 'off') {
            if (api === 'openai-responses') delete merged.reasoning;
            else if (api === 'openai-completions') delete merged.reasoning_effort;
          }
          if (api === 'google-generative-ai') delete merged.temperature;
          if (effective.toolChoice != null) {
            if (api === 'anthropic-messages') merged.tool_choice = { type: effective.toolChoice };
            else if (api === 'google-generative-ai') merged.config.toolConfig = { ...merged.config.toolConfig, functionCallingConfig: { mode: effective.toolChoice === 'none' ? 'NONE' : 'AUTO' } };
            else if (['openai-responses', 'openai-completions'].includes(api)) merged.tool_choice = effective.toolChoice;
            else throw error('UNSUPPORTED_TOOL_CHOICE', `toolChoice is not configured for ${api}`);
          }
          checkRequestAdmission(merged, capability, api);
          return merged;
        };
        if (api === 'google-generative-ai') {
          installGoogleTransport();
          // The official adapter checks identity against globalThis.fetch.
          effective.fetch = globalThis.fetch;
          const base = new URL(provider.base_url);
          const scope = { fetch: transport, matches(input) {
            const url = new URL(input instanceof Request ? input.url : String(input));
            return url.origin === base.origin && url.pathname.startsWith(`${base.pathname.replace(/\/$/, '')}/`)
              && url.pathname.endsWith(':streamGenerateContent');
          } };
          return googleTransport.run(scope, () => streamSimple(model, context, effective));
        }
        effective.fetch = transport;
        return streamSimple(model, context, effective);
      },
    });
    // AuthStorage is in-memory. Dummy auth supports authenticated-by-header and local services.
    await runtime.setRuntimeApiKey(id, provider.api_key || 'maibot-no-auth');
  }
  return runtime;
}

export function getHostRuntimeState(runtime) {
  const state = states.get(runtime);
  if (!state) throw error('INVALID_RUNTIME', 'Not a host-configured Pi ModelRuntime');
  return state;
}

/** Selection is host policy, execution/replay remains Pi. No fallback to a different model on error. */
export function resolveTaskModel(runtime, taskName, { modelOverride, options = {} } = {}) {
  const state = getHostRuntimeState(runtime);
  options = structuredClone(options);
  const task = state.config.model_task_config?.[taskName];
  if (!task && !modelOverride) throw error('UNKNOWN_MODEL_TASK', `Unknown model task: ${taskName}`);
  const names = modelOverride ? [modelOverride] : task.model_list;
  if (!names?.length) throw error('EMPTY_MODEL_TASK', `No configured model for ${taskName}`);
  const strategy = task?.selection_strategy ?? 'sequential';
  let name;
  if (strategy === 'random') name = names[Math.floor(Math.random() * names.length)];
  else if (strategy === 'balance') name = [...names].sort((a, b) => (state.counts.get(a) ?? 0) - (state.counts.get(b) ?? 0))[0];
  else if (strategy === 'sequential') name = names[0];
  else throw error('UNSUPPORTED_MODEL_SELECTION', `Unknown selection strategy: ${strategy}`);
  const route = state.routes.get(name);
  if (!route) throw error('UNKNOWN_HOST_MODEL', `Unknown host model: ${name}`);
  const { info, provider, extra, capability } = route;
  const override = splitExtraParams(options.extraParams);
  // Request snapshots can change sampling/body values, never silently replace
  // the configured transport or capability policy. Repeated config values are OK.
  for (const key of ['headers', 'query', 'pi']) for (const [name, value] of Object.entries(override[key])) {
    if (canonical(value) !== canonical(extra[key][name])) throw error('RESERVED_ROUTE_PARAMETER', `Request extraParams cannot replace configured ${key}.${name}`);
  }
  const body = options.extraParamsReplace === true ? override.body : { ...extra.body, ...override.body };
  validateExtraBody(body);
  const registered = route.providerId && runtime.getModel(route.providerId, info.model_identifier);
  if (!registered) throw error('UNSUPPORTED_CHAT_PROVIDER', `Model ${name} is not configured as a supported Pi chat provider`);
  state.counts.set(name, (state.counts.get(name) ?? 0) + 1);
  const maxTokens = options.max_tokens ?? options.maxTokens ?? info.max_tokens ?? configuredOutput(body) ?? task?.max_tokens ?? registered.maxTokens;
  if (!positive(maxTokens)) throw error('INVALID_OUTPUT_LIMIT', 'Output budget must be a positive integer');
  const thinkingLevel = options.thinkingLevel ?? body.reasoning_effort ?? body.reasoning?.effort ?? extra.pi.thinkingLevel ?? 'off';
  if (!LEVELS.includes(thinkingLevel)) throw error('UNSUPPORTED_REASONING', 'Unknown reasoning effort');
  const temperature = info.send_temperature === false ? undefined : (options.temperature ?? info.temperature ?? body.temperature ?? body.config?.temperature ?? body.generationConfig?.temperature ?? task?.temperature);
  const toolChoice = options.toolChoice ?? body.tool_choice;
  if (toolChoice != null && !['auto', 'none'].includes(toolChoice)) throw error('UNSUPPORTED_TOOL_CHOICE', 'Pi host toolChoice must be auto or none');
  const requestOptions = { maxTokens, ...(temperature == null ? {} : { temperature }), ...(toolChoice == null ? {} : { toolChoice }), extraParams: body };
  const model = { ...registered, maxTokens, reasoning: thinkingLevel !== 'off', hostExtraBody: body, hostRequestOptions: requestOptions };
  const routeFingerprint = fingerprint({ toolChoice, temperature, sendTemperature: info.send_temperature, capability, compatibility: extra.pi.compat, credentialDigest: fingerprint(provider.api_key ?? ''), authHeaderName: provider.auth_header_name, authHeaderPrefix: provider.auth_header_prefix, authQueryName: provider.auth_query_name, organization: provider.organization, project: provider.project, provider: info.api_provider, endpoint: provider.base_url, model: info.model_identifier, api: route.api, thinkingLevel, maxTokens, headers: { ...provider.default_headers, ...extra.headers }, query: { ...provider.default_query, ...extra.query }, body, authType: provider.auth_type ?? 'bearer' });
  return { model, thinkingLevel, maxTokens, temperature, requestOptions, capability, hostModelName: name, apiProvider: info.api_provider,
    warnings: capability.contextKnown ? [] : ['Context window is unverified; audit mode does not establish a hard input/output safety bound.'], timeoutMs: (task?.hard_timeout ?? provider.timeout ?? 240) * 1000, routeFingerprint };
}
