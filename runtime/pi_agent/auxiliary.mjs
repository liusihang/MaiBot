/** Explicit Pi-side non-chat adapters. Pi 1.4 has no embedding or audio operation.
 * These adapters execute in Node (never Python), with shared route/auth/cancellation.
 * They are not represented as fabricated assistant reasoning or chat completions.
 */
import { createHash } from 'node:crypto';
import { checkRequestAdmission, createProviderFetch, getHostRuntimeState, splitExtraParams } from './provider.mjs';
const fail = (message) => Object.assign(new Error(message), { code: 'UNSUPPORTED_MODEL_OPERATION' });
const canonical = (value) => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b))) : v);
const hash = (value) => createHash('sha256').update(canonical(value)).digest('hex');
const endpoint = (base, suffix) => `${base.replace(/\/$/, '')}/${suffix}`;

function selectRoute(runtime, payload) {
  const state = getHostRuntimeState(runtime);
  const task = state.config.model_task_config?.[payload.taskName];
  const names = payload.modelOverride ? [payload.modelOverride] : task?.model_list;
  if (!names?.length) throw fail(`No configured auxiliary model for ${payload.taskName}`);
  const strategy = task?.selection_strategy ?? 'sequential';
  let name;
  if (strategy === 'random') name = names[Math.floor(Math.random() * names.length)];
  else if (strategy === 'balance') name = [...names].sort((a, b) => (state.counts.get(a) ?? 0) - (state.counts.get(b) ?? 0))[0];
  else if (strategy === 'sequential') name = names[0];
  else throw fail(`Unknown selection strategy: ${strategy}`);
  const route = state.routes.get(name);
  if (!route) throw fail(`Unknown auxiliary model: ${name}`);
  state.counts.set(name, (state.counts.get(name) ?? 0) + 1);
  return { ...route, task, fetchImpl: state.fetchImpl };
}
function vector(raw) {
  if (!Array.isArray(raw) || !raw.length || raw.some((n) => typeof n !== 'number' || !Number.isFinite(n))) throw fail('Embedding response is not a non-empty finite vector');
  return raw;
}
function media(input) {
  if (!input || typeof input.data !== 'string' || !input.data || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(input.data)) throw fail('Expected base64 media input');
  return { data: input.data, mimeType: input.mimeType ?? 'audio/wav' };
}
function render(value, replacements) {
  if (typeof value === 'string') return value.replace(/\{(data_uri|image_base64|mime_type|model)\}/g, (_, key) => replacements[key]);
  if (Array.isArray(value)) return value.map((item) => render(item, replacements));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, render(item, replacements)]));
  return value;
}
function imageProtocol(provider, pi) {
  if (pi.imageEmbeddingProtocol) return pi.imageEmbeddingProtocol;
  if (provider.client_type === 'qwen3vl_image') return 'qwen3vl-retrieval-v1';
  const url = new URL(provider.base_url);
  const path = url.pathname.replace(/\/$/, '');
  if ((url.hostname === 'dashscope.aliyuncs.com' || /^[a-z0-9-]+\.[a-z0-9-]+\.maas\.aliyuncs\.com$/.test(url.hostname)) && ['/compatible-mode/v1', '/api/v1'].includes(path)) return 'dashscope';
  if (url.hostname === 'ark.cn-beijing.volces.com' && ['/api/v3', '/api/plan/v3', '/api/coding/v3'].includes(path)) return 'ark';
  if (['api.siliconflow.cn', 'api.siliconflow.com'].includes(url.hostname) && path === '/v1') return 'siliconflow';
  return null;
}

export async function runAuxiliaryOperation(runtime, operation, payload, { signal } = {}) {
  signal?.throwIfAborted();
  const selected = selectRoute(runtime, payload);
  const { info, provider, task, fetchImpl } = selected;
  // Embedding dimensions, audio input and generated images do not have the
  // chat-token semantics required by this admission contract. Audit keeps the
  // explicit adapters usable; strict must not silently claim a proof we lack.
  if (selected.capability.mode === 'strict') throw Object.assign(new Error(`Strict capacity admission cannot establish a safe ${operation} operation bound`), { code: 'CONTEXT_CAPABILITY_UNKNOWN' });
  const defaults = splitExtraParams(info.extra_params);
  const overrides = splitExtraParams(payload.options?.extraParams);
  for (const key of ['headers', 'query', 'pi']) for (const [name, value] of Object.entries(overrides[key])) {
    if (canonical(value) !== canonical(defaults[key][name])) throw fail(`Request extraParams cannot replace configured ${key}.${name}`);
  }
  const extra = { headers: defaults.headers, query: defaults.query, pi: defaults.pi, body: payload.options?.extraParamsReplace === true ? overrides.body : { ...defaults.body, ...overrides.body } };
  const timeoutMs = Math.min(task?.hard_timeout ?? 240, provider.timeout ?? 240) * 1000;
  const effectiveSignal = AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(timeoutMs)]);
  const fetcher = createProviderFetch(provider, extra, fetchImpl);
  if (operation === 'generate_image') {
    const imageModel = selected.providerId && runtime.getModelOfType('image', selected.providerId, info.model_identifier);
    if (!imageModel) throw fail('Image generation requires extra_params.pi.operation=image with an image provider API');
    const input = typeof payload.input === 'string' ? [{ type: 'text', text: payload.input }] : payload.input?.input ?? [
      { type: 'text', text: payload.input?.prompt ?? '' },
      ...(payload.input?.images ?? []).map((image) => {
        const decoded = media(image);
        if (!decoded.mimeType.startsWith('image/')) throw fail('Image generation requires an image MIME type');
        return { type: 'image', ...decoded };
      }),
    ];
    const { extraParams, extraParamsReplace, ...imageOptions } = payload.options ?? {};
    const message = await runtime.generateImages(imageModel, { input }, { signal: effectiveSignal, fetch: fetcher, metadata: { ...extra.body, ...imageOptions } });
    if (message.stopReason !== 'stop') { effectiveSignal.throwIfAborted(); throw fail(message.errorMessage || 'Pi image generation failed'); }
    const route = { modelName: info.name, modelIdentifier: info.model_identifier, apiProvider: info.api_provider, operation, nativePiOperation: true };
    return { message, output: message.output, route, ...route, usage: message.usage ?? null };
  }
  const params = { ...extra.body };
  for (const key of ['model', 'input', 'messages', 'contents', 'file', 'encoding_format']) {
    if (key in params) throw fail(`Auxiliary parameters cannot override identity/input field: ${key}`);
  }
  if (payload.options?.dimensions != null) params.dimensions = payload.options.dimensions;
  const requestedDimensions = params.dimensions ?? params.output_dimensionality ?? params.outputDimensionality ?? params.parameters?.dimension;
  if (requestedDimensions != null && (!Number.isSafeInteger(requestedDimensions) || requestedDimensions <= 0)) throw fail('Embedding dimensions must be a positive integer');
  const route = { modelName: info.name, modelIdentifier: info.model_identifier, apiProvider: info.api_provider, operation, nativePiOperation: false };
  const headers = { 'content-type': 'application/json' };
  if (provider.api_key && (provider.auth_type ?? 'bearer') === 'bearer') headers.authorization = `Bearer ${provider.api_key}`;
  let url, body, protocol, parse;
  const model = info.model_identifier;
  if (operation === 'embed_text') {
    if (info.model_type !== 'embedding') throw fail(`Model ${info.name} is not a text embedding model`);
    if (typeof payload.input !== 'string') throw fail('Text embedding input must be a string');
    if (provider.client_type === 'gemini') {
      protocol = 'google-embedContent';
      url = endpoint(provider.base_url, `models/${encodeURIComponent(model.replace(/^models\//, ''))}:embedContent`);
      headers['x-goog-api-key'] = provider.api_key; delete headers.authorization;
      const { dimensions, output_dimensionality, outputDimensionality, task_type, ...rest } = params;
      body = { ...rest, model: `models/${model.replace(/^models\//, '')}`, content: { parts: [{ text: payload.input }] },
        ...(requestedDimensions ? { outputDimensionality: requestedDimensions } : {}), ...(task_type ? { taskType: task_type } : {}) };
      parse = (raw) => ({ embedding: vector(raw.embedding?.values) });
    } else if (['openai', 'openai_responses'].includes(provider.client_type)) {
      protocol = 'openai-embeddings'; url = endpoint(provider.base_url, 'embeddings');
      body = { ...params, model, input: payload.input, encoding_format: 'float' };
      parse = (raw) => ({ embedding: vector(raw.data?.[0]?.embedding) });
    } else throw fail(`No Pi-side embedding adapter for ${provider.client_type}`);
  } else if (operation === 'embed_image') {
    if (info.model_type !== 'image_embedding') throw fail(`Model ${info.name} is not an image embedding model`);
    const input = media(payload.input);
    if (!input.mimeType.startsWith('image/')) throw fail('Image embedding requires an image MIME type');
    const dataUri = `data:${input.mimeType};base64,${input.data}`;
    protocol = imageProtocol(provider, extra.pi);
    const { image_embedding_input, image_embedding_body, ...rest } = params;
    if (image_embedding_input || image_embedding_body) {
      protocol = 'openai-image-template'; url = endpoint(provider.base_url, 'embeddings');
      const replacements = { data_uri: dataUri, image_base64: input.data, mime_type: input.mimeType, model };
      body = { ...rest, ...render(image_embedding_body ?? {}, replacements), model, encoding_format: 'float' };
      if (image_embedding_input) body.input = render(image_embedding_input, replacements);
      parse = (raw) => ({ embedding: vector(raw.data?.[0]?.embedding) });
    } else if (protocol === 'qwen3vl-retrieval-v1') {
      url = endpoint(provider.base_url, 'embeddings');
      body = { ...rest, model, messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: dataUri } }] }], encoding_format: 'float' };
      parse = (raw) => ({ embedding: vector(raw.data?.[0]?.embedding) });
    } else if (protocol === 'dashscope') {
      url = new URL('/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding', provider.base_url).href;
      const { dimensions, parameters = {}, ...other } = rest;
      if (['model', 'input'].some((key) => key in other)) throw fail('Image embedding reserved fields cannot be overridden');
      body = { ...other, model, input: { contents: [{ image: dataUri }] }, parameters: { ...parameters, ...(dimensions ? { dimension: dimensions } : {}) } };
      parse = (raw) => ({ embedding: vector(raw.output?.embeddings?.[0]?.embedding) });
    } else if (protocol === 'ark') {
      url = endpoint(provider.base_url, 'embeddings/multimodal');
      body = { ...rest, model, input: [{ type: 'image_url', image_url: { url: dataUri } }], encoding_format: 'float' };
      parse = (raw) => ({ embedding: vector(raw.data?.embedding) });
    } else if (protocol === 'siliconflow') {
      url = endpoint(provider.base_url, 'embeddings'); body = { ...rest, model, input: { image: dataUri }, encoding_format: 'float' };
      parse = (raw) => ({ embedding: vector(raw.data?.[0]?.embedding) });
    } else throw fail('Image embedding requires a verified protocol or explicit image_embedding_input/body template');
  } else if (operation === 'transcribe') {
    const input = media(payload.input);
    if (!input.mimeType.startsWith('audio/')) throw fail('Transcription requires an audio MIME type');
    if (['openai', 'openai_responses'].includes(provider.client_type)) {
      protocol = 'openai-audio-transcriptions'; url = endpoint(provider.base_url, 'audio/transcriptions');
      body = new FormData(); body.set('model', model);
      body.set('file', new Blob([Buffer.from(input.data, 'base64')], { type: input.mimeType }), `audio.${input.mimeType.split('/')[1] || 'wav'}`);
      for (const [key, value] of Object.entries(params)) body.set(key, typeof value === 'string' ? value : JSON.stringify(value));
      delete headers['content-type']; parse = (raw) => {
        const text = typeof raw === 'string' ? raw : raw.text;
        if (typeof text !== 'string') throw fail('Audio transcription returned no text');
        return { text };
      };
    } else if (provider.client_type === 'gemini') {
      protocol = 'google-audio-transcription'; url = endpoint(provider.base_url, `models/${encodeURIComponent(model.replace(/^models\//, ''))}:generateContent`);
      headers['x-goog-api-key'] = provider.api_key; delete headers.authorization;
      const { generationConfig = {}, max_tokens, max_output_tokens, ...flat } = params;
      const outputBudget = payload.options?.maxTokens ?? payload.options?.max_tokens ?? info.max_tokens ?? max_output_tokens ?? max_tokens ?? task?.max_tokens;
      const sampling = Object.fromEntries(Object.entries(flat).map(([key, value]) => [key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()), value]));
      body = { contents: [{ role: 'user', parts: [{ text: 'Transcribe the audio faithfully. Return only the transcription.' }, { inlineData: { mimeType: input.mimeType, data: input.data } }] }],
        generationConfig: { ...sampling, ...generationConfig, ...(outputBudget == null ? {} : { maxOutputTokens: outputBudget }) } };
      checkRequestAdmission(body, selected.capability, 'google-audio-transcription');
      parse = (raw) => {
        const text = raw.candidates?.[0]?.content?.parts?.filter((part) => part.text != null).map((part) => part.text).join('');
        if (typeof text !== 'string') throw fail('Audio transcription returned no text');
        return { text };
      };
    } else throw fail(`No Pi-side ASR adapter for ${provider.client_type}`);
  } else throw fail(`Unknown auxiliary operation: ${operation}`);
  const response = await fetcher(url, { method: 'POST', headers, body: body instanceof FormData ? body : JSON.stringify(body), signal: effectiveSignal });
  if (!response.ok) { await response.body?.cancel(); throw Object.assign(new Error(`Pi auxiliary ${protocol} HTTP ${response.status}`), { code: 'MODEL_HTTP_ERROR', status: response.status }); }
  const raw = response.headers.get('content-type')?.includes('json') ? await response.json() : await response.text();
  effectiveSignal.throwIfAborted();
  const result = parse(raw);
  if (result.embedding && requestedDimensions != null && result.embedding.length !== requestedDimensions) throw fail('Embedding response dimension does not match the requested dimension');
  // Identity excludes input bytes and credentials, and includes the protocol/preprocessing contract.
  const requestProtocolHash = hash({ protocol, endpoint: url, model, params, actualDimensions: result.embedding?.length ?? null,
    headers: { ...provider.default_headers, ...extra.headers }, query: { ...provider.default_query, ...extra.query },
    credentialDigest: hash(provider.api_key ?? ''), authType: provider.auth_type ?? 'bearer', authHeaderName: provider.auth_header_name,
    authHeaderPrefix: provider.auth_header_prefix, authQueryName: provider.auth_query_name, organization: provider.organization, project: provider.project,
    mimeType: payload.input?.mimeType ?? null, preprocessVersion: payload.input?.preprocessVersion ?? null });
  return { ...result, route, ...route, requestProtocolHash, usage: typeof raw === 'object' ? (raw.usage ?? raw.usageMetadata ?? null) : null };
}
