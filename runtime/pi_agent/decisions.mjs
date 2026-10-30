/** Participation only, through Pi's native OpenAI Decisions classifier.
 * No assistant transcript, tools, reply plan, alternate provider, or retry.
 */
import { createHash } from 'node:crypto';
import { fetch as proxyFetch, ProxyAgent } from 'undici';
import { selectAuxiliaryRoute } from './auxiliary.mjs';
import { createProviderFetch } from './provider.mjs';

const NAME = 'group_participation';
const CHOICES = ['engage', 'ignore'];
const record = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const fail = (code, message, details) => Object.assign(new Error(message), { code,
  ...(details ? { safeDiagnostics: details } : {}) });
const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const probability = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
// The alias is shared with chat/memory. Their sampling/output fields have no
// Decisions meaning and are deliberately NOT copied into its request body.
const CHAT_ONLY = new Set(['reasoning', 'reasoning_effort', 'temperature', 'top_p', 'max_tokens',
  'max_output_tokens', 'max_completion_tokens', 'stream', 'store', 'tool_choice', 'response_format']);
const CACHE_FIELDS = new Set(['prompt_cache_key', 'prompt_cache_options', 'safety_identifier']);
const OPTION_FIELDS = new Set(['promptCacheKey', 'promptCacheOptions', 'safetyIdentifier', 'timeoutMs']);

function decisionProxy(extra) {
  const address = extra.pi.decisionsProxyUrl;
  if (address == null) return undefined;
  let url;
  try { url = new URL(address); } catch { /* Report only a safe code below. */ }
  if (!url || !['http:', 'https:'].includes(url.protocol)) {
    throw fail('INVALID_DECISION_PROXY', 'Decisions proxy must be an HTTP or HTTPS URL');
  }
  return new ProxyAgent({ uri: url.href, proxyTunnel: true });
}

function validateInput(input) {
  if (!record(input)) throw fail('INVALID_DECISION_INPUT', 'Participation evidence must be a JSON object');
  let values = 0;
  const visit = (value, depth) => {
    if (++values > 10000) throw fail('INVALID_DECISION_INPUT', 'Decisions JSON evidence exceeds 10000 values');
    if (value !== null && typeof value === 'object') {
      if (depth > 32) throw fail('INVALID_DECISION_INPUT', 'Decisions JSON evidence exceeds 32 container levels');
      for (const child of Object.values(value)) visit(child, depth + 1);
    } else if (!['string', 'number', 'boolean'].includes(typeof value) && value !== null
        || typeof value === 'number' && !Number.isFinite(value)) {
      throw fail('INVALID_DECISION_INPUT', 'Participation evidence contains a non-JSON value');
    }
  };
  visit(input, 1);
  const serialized = JSON.stringify(input);
  if (Buffer.byteLength(serialized) > 256 * 1024) {
    throw fail('INVALID_DECISION_INPUT', 'Decisions JSON evidence exceeds 256 KiB; source messages were not truncated');
  }
  return JSON.parse(serialized);
}

function requestOptions(extra, options, sessionId) {
  if (!record(options) || Object.keys(options).some((key) => !OPTION_FIELDS.has(key))) {
    throw fail('INVALID_DECISION_OPTIONS', 'Unsupported Decisions request option');
  }
  const body = {};
  for (const [key, value] of Object.entries(extra.body)) {
    if (CACHE_FIELDS.has(key)) body[key] = value;
    else if (!CHAT_ONLY.has(key)) throw fail('INVALID_DECISION_OPTIONS', 'Configured model body contains an unsupported Decisions parameter');
  }
  for (const [option, wire] of [['promptCacheKey', 'prompt_cache_key'], ['promptCacheOptions', 'prompt_cache_options'], ['safetyIdentifier', 'safety_identifier']]) {
    if (options[option] !== undefined) body[wire] = options[option];
  }
  body.prompt_cache_key ??= `mai:decision:${createHash('sha256').update(String(sessionId ?? '')).digest('hex').slice(0, 48)}`;
  for (const [key, max] of [['prompt_cache_key', 64], ['safety_identifier', 128]]) {
    if (body[key] != null && (typeof body[key] !== 'string' || [...body[key]].length > max)) {
      throw fail('INVALID_DECISION_OPTIONS', 'Invalid Decisions cache key or safety identifier');
    }
  }
  if (body.prompt_cache_options != null) {
    const cache = body.prompt_cache_options;
    if (!record(cache) || Object.keys(cache).some((key) => !['mode', 'ttl'].includes(key))
        || cache.mode != null && !['implicit', 'explicit'].includes(cache.mode)
        || 'ttl' in cache && cache.ttl !== '30m') {
      throw fail('INVALID_DECISION_OPTIONS', 'Decisions caching supports implicit/explicit mode and 30m TTL only');
    }
  }
  return body;
}

function usageReceipt(raw) {
  if (!record(raw) || !integer(raw.input_tokens) || !integer(raw.output_tokens) || !integer(raw.total_tokens)
      || !record(raw.input_tokens_details) || !integer(raw.input_tokens_details.cached_tokens)
      || !integer(raw.input_tokens_details.cache_write_tokens) || !record(raw.output_tokens_details)
      || !integer(raw.output_tokens_details.reasoning_tokens)) return null;
  const cacheRead = raw.input_tokens_details.cached_tokens;
  const cacheWrite = raw.input_tokens_details.cache_write_tokens;
  if (cacheRead + cacheWrite > raw.input_tokens || raw.total_tokens !== raw.input_tokens + raw.output_tokens
      || raw.output_tokens_details.reasoning_tokens > raw.output_tokens) return null;
  return { input: raw.input_tokens - cacheRead - cacheWrite, output: raw.output_tokens,
    cacheRead, cacheWrite, cacheReported: true, totalTokens: raw.total_tokens };
}

function validateAnswer(raw) {
  if (!record(raw) || raw.model !== 'gpt-6-luna' || !Array.isArray(raw.answers) || raw.answers.length !== 1) {
    throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned an invalid model or answer count');
  }
  const answer = raw.answers[0];
  if (!record(answer) || answer.name !== NAME) throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned an invalid question identity');
  if (answer.type === 'refusal') throw fail('DECISION_REFUSED', 'Decisions declined the participation question');
  if (answer.type !== 'choice' || !CHOICES.includes(answer.choice) || !probability(answer.confidence)
      || !Array.isArray(answer.probabilities) || answer.probabilities.length !== CHOICES.length) {
    throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned an invalid participation choice');
  }
  const seen = new Set();
  for (const item of answer.probabilities) {
    if (!record(item) || !CHOICES.includes(item.value) || seen.has(item.value) || !probability(item.probability)) {
      throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned invalid choice probabilities');
    }
    seen.add(item.value);
  }
  return answer;
}

export async function decideParticipation(runtime, payload, { signal } = {}) {
  signal?.throwIfAborted();
  const input = validateInput(payload.input);
  if (typeof payload.instructions !== 'string' || !payload.instructions.trim()
      || payload.instructions.length > 1048576) throw fail('INVALID_DECISION_INPUT', 'Decisions requires participation instructions of at most 1048576 characters');
  const selected = selectAuxiliaryRoute(runtime, payload);
  const { info, provider, extra, capability, task, fetchImpl } = selected;
  if (info.model_identifier !== 'gpt-6-luna' || !selected.classifierProviderId) {
    throw fail('UNSUPPORTED_DECISION_MODEL', 'Decisions requires an OpenAI-compatible gpt-6-luna route');
  }
  // No chat-output token budget applies to a classifier. Audit is explicit;
  // strict must not claim a bound that the existing chat counter cannot prove.
  if (capability.mode === 'strict') throw fail('CONTEXT_CAPABILITY_UNKNOWN', 'Strict admission cannot prove a Decisions operation bound');
  const options = payload.options ?? {};
  const fields = requestOptions(extra, options, payload.sessionId);
  const timeoutMs = Math.min(options.timeoutMs ?? 30000, (task?.hard_timeout ?? 240) * 1000, (provider.timeout ?? 240) * 1000);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2147483647) throw fail('INVALID_DECISION_OPTIONS', 'Decisions timeout must be positive milliseconds');
  const effectiveSignal = AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(timeoutMs)]);
  const route = { modelName: info.name, modelIdentifier: info.model_identifier, apiProvider: info.api_provider,
    operation: 'decide', nativePiOperation: true };
  const started = performance.now();
  const details = { route };
  let wireError;
  let proxy;
  // Only this classification uses the configured egress. Other model tasks
  // and process-wide HTTP settings keep their existing routes.
  const transport = (url, init) => (fetchImpl ?? (proxy ? proxyFetch : globalThis.fetch))(url, {
    ...init, ...(proxy ? { dispatcher: proxy } : {}),
  });
  const fetcher = createProviderFetch(provider, extra, transport);
  const guardedFetch = async (url, init) => {
    try {
      const response = await fetcher(url, init);
      if (!response.ok) {
        details.httpStatus = response.status;
        await response.body?.cancel();
        throw fail('DECISION_HTTP_ERROR', `Decisions HTTP ${response.status}`);
      }
      let raw;
      try { raw = await response.clone().json(); }
      catch { await response.body?.cancel(); throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned invalid JSON'); }
      const usage = usageReceipt(raw?.usage);
      if (usage) {
        details.usage = usage;
        // Whitelist token fields; never echo arbitrary provider diagnostic data.
        details.providerUsage = { input_tokens: raw.usage.input_tokens, output_tokens: raw.usage.output_tokens,
          total_tokens: raw.usage.total_tokens,
          input_tokens_details: { cached_tokens: usage.cacheRead, cache_write_tokens: usage.cacheWrite },
          output_tokens_details: { reasoning_tokens: raw.usage.output_tokens_details.reasoning_tokens } };
      }
      try {
        validateAnswer(raw);
        if (!usage) throw fail('INVALID_DECISION_RESPONSE', 'Decisions returned invalid token usage');
      } catch (error) { await response.body?.cancel(); throw error; }
      return response;
    } catch (error) {
      // Pi serializes provider exceptions into errorMessage. Keep a safe typed
      // error outside that projection, never expose a response body/URL/key.
      wireError = error?.code && ['DECISION_HTTP_ERROR', 'INVALID_DECISION_RESPONSE', 'DECISION_REFUSED'].includes(error.code)
        ? error : fail('DECISION_TRANSPORT_ERROR', 'Decisions transport did not complete');
      throw wireError;
    }
  };
  const model = runtime.getModelOfType('classifier', selected.classifierProviderId, info.model_identifier);
  let result;
  proxy = decisionProxy(extra);
  try {
    result = await runtime.classify(model, {
      state: input,
      questions: { [NAME]: { type: 'choice', instructions: payload.instructions,
        criteria: Object.fromEntries(CHOICES.map((choice) => [choice, ''])) } },
    }, {
      signal: effectiveSignal, maxRetries: 0, timeoutMs, fetch: guardedFetch,
      // Pi 1.4 native Decisions predates GA input_json/cache support. Use its
      // official payload hook, keeping all participant-supplied keys as DATA.
      onPayload: (body) => ({ ...body, ...fields, input: [{ role: 'user', content: [{ type: 'input_json', data: input }] }] }),
    });
  } finally {
    // Classification has consumed the response, or was cancelled. Never keep
    // request-scoped CONNECT sockets alive after the operation ends.
    await proxy?.destroy();
  }
  details.durationMs = performance.now() - started;
  if (signal?.aborted) throw fail('DECISION_CANCELLED', 'Decisions participation request was cancelled', details);
  if (effectiveSignal.aborted) throw fail('DECISION_TIMEOUT', 'Decisions participation request timed out', details);
  if (wireError) throw fail(wireError.code, wireError.message, details);
  if (result.stopReason !== 'stop') throw fail('DECISION_PROVIDER_ERROR', 'Pi Decisions classifier did not complete', details);
  const answer = result.answers[NAME];
  // Both raw validation and native parsing must agree before admission.
  if (!answer || !CHOICES.includes(answer.choice)) throw fail('INVALID_DECISION_RESPONSE', 'Pi returned an invalid participation choice', details);
  return { decision: answer.choice, confidence: answer.confidence, probabilities: answer.probabilities, ...details };
}
