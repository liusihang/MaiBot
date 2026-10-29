import assert from 'node:assert/strict';
import test from 'node:test';
import { createHostModelRuntime, resolveTaskModel, checkRequestAdmission } from '../provider.mjs';

const config = (extra = {}) => ({
  api_providers: [{ name: 'gateway', client_type: 'openai_responses', base_url: 'https://unit.invalid/v1', api_key: 'test-secret', default_headers: { 'X-Lite': 'true' } }],
  models: [{ name: 'main', model_identifier: 'exact-terra-route', model_type: 'llm', api_provider: 'gateway', max_tokens: 25600, visual: true, extra_params: { reasoning_effort: 'high', reasoning: { context: 'preserve' }, ...extra } },
    { name: 'helper', model_identifier: 'exact-luna-route', model_type: 'llm', api_provider: 'gateway', extra_params: { reasoning_effort: 'max' } }],
  model_task_config: { main: { model_list: ['main'], max_tokens: 1024, temperature: 0.7 }, helper: { model_list: ['helper'], max_tokens: 4096 } },
});
const response = () => new Response([
  'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1","model":"exact-terra-route","status":"in_progress","output":[]}}\n\n',
  'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","model":"exact-terra-route","status":"completed","output":[],"usage":{"input_tokens":4,"output_tokens":0,"total_tokens":4}}}\n\n',
].join(''), { headers: { 'content-type': 'text/event-stream' } });

test('official Responses provider serializes exact effort/output/extra body/headers with unknown capacity explicit', async () => {
  const requests = [];
  const runtime = await createHostModelRuntime(config(), { fetch: async (url, init) => { requests.push({ url: String(url), init }); return response(); } });
  const route = resolveTaskModel(runtime, 'main');
  assert.equal(route.thinkingLevel, 'high');
  assert.equal(route.maxTokens, 25600);
  assert.equal(route.capability.contextKnown, false);
  assert.equal(resolveTaskModel(runtime, 'helper').thinkingLevel, 'max');
  const result = await runtime.completeSimple(route.model, { messages: [{ role: 'user', content: 'hello', timestamp: 1 }] }, { reasoning: route.thinkingLevel });
  assert.notEqual(result.stopReason, 'error', result.errorMessage);
  const wire = JSON.parse(requests[0].init.body);
  assert.equal(wire.max_output_tokens, 25600);
  assert.deepEqual(wire.reasoning, { effort: 'high', summary: 'auto', context: 'preserve' });
  assert.equal(new Headers(requests[0].init.headers).get('X-Lite'), 'true');
  assert.equal(new Headers(requests[0].init.headers).get('authorization'), 'Bearer test-secret');
  assert.equal(wire.model, 'exact-terra-route');
  assert.equal(wire.input[0].content[0].text, 'hello');
});

test('strict unknown refuses, audit unknown does not pretend exact, declared bound rejects overflow', () => {
  assert.throws(() => checkRequestAdmission({ max_output_tokens: 20 }, { mode: 'strict' }, 'openai-responses'), /cannot establish/);
  assert.equal(checkRequestAdmission({ max_output_tokens: 20 }, { mode: 'audit' }, 'openai-responses').verified, false);
  assert.throws(() => checkRequestAdmission({ input: 'x'.repeat(100), max_output_tokens: 20 }, { mode: 'audit', contextKnown: true, contextWindow: 100, output_semantics: 'includes_reasoning', input_counter: 'utf8_bytes_bound', tokens_per_utf8_byte_upper: 1, protocol_overhead_tokens: 0 }, 'openai-responses'), /exceeds/);
});

test('unsupported plugin model fails explicitly without fallback', async () => {
  const cfg = config(); cfg.api_providers[0].client_type = 'unknown-plugin';
  const runtime = await createHostModelRuntime(cfg);
  assert.throws(() => resolveTaskModel(runtime, 'main'), /not configured as a supported Pi/);
});

test('Pi estimate clamp does not silently reduce the configured final Responses output allowance', async () => {
  const cfg = config(); cfg.models[0].request_admission = { mode: 'audit', context_window: 27000 };
  let wire;
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { wire = JSON.parse(init.body); return response(); } });
  const route = resolveTaskModel(runtime, 'main');
  const result = await runtime.completeSimple(route.model, { messages: [{ role: 'user', content: 'x'.repeat(16000), timestamp: 1 }] }, { reasoning: route.thinkingLevel });
  assert.notEqual(result.stopReason, 'error', result.errorMessage);
  assert.equal(wire.max_output_tokens, 25600);
});

test('route header values affect replay identity and reserved model rewrite fails closed', async () => {
  const first = config(); const second = config(); second.api_providers[0].default_headers['X-Lite'] = 'false';
  const a = resolveTaskModel(await createHostModelRuntime(first), 'main');
  const b = resolveTaskModel(await createHostModelRuntime(second), 'main');
  assert.notEqual(a.routeFingerprint, b.routeFingerprint);
  await assert.rejects(createHostModelRuntime(config({ model: 'different-model' })), /cannot override/);
});

const googleResponse = (text = 'google') => new Response(`data: ${JSON.stringify({ candidates: [{ content: { role: 'model', parts: [{ text, thoughtSignature: 'opaque-google-signature' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 2, totalTokenCount: 6 } })}\n\n`, { headers: { 'content-type': 'text/event-stream' } });
const anthropicResponse = () => new Response([
  { type: 'message_start', message: { id: 'msg_fixture', type: 'message', role: 'assistant', model: 'exact-terra-route', content: [], stop_reason: null, usage: { input_tokens: 4, output_tokens: 0 } } },
  { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'anthropic' } },
  { type: 'content_block_stop', index: 0 },
  { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 2 } },
  { type: 'message_stop' },
].map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });
const complete = (runtime, route, options = {}) => runtime.completeSimple(route.model,
  { messages: [{ role: 'user', content: 'x'.repeat(16000), timestamp: 1 }] }, { reasoning: route.thinkingLevel, ...options });

for (const [type, responseFactory, outputField] of [
  ['gemini', googleResponse, 'maxOutputTokens'], ['anthropic', anthropicResponse, 'max_tokens'],
  ['openai', () => new Response('data: {"id":"chat-1","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } }), 'max_completion_tokens'],
]) test(`official ${type} final wire budget is exact after estimation and hooks`, async () => {
  const cfg = config(); cfg.api_providers[0].client_type = type;
  cfg.models[0].request_admission = { mode: 'audit', context_window: 27000 };
  cfg.models[0].extra_params = type === 'gemini' ? { generationConfig: { topP: 0.8, maxOutputTokens: 999999 } } : { top_p: 0.8, max_completion_tokens: 999999 };
  let wire;
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { wire = JSON.parse(init.body); return responseFactory(); } });
  const route = resolveTaskModel(runtime, 'main');
  const result = await complete(runtime, route, { onPayload: (payload) => ({ ...payload, max_output_tokens: 999999 }) });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  const sampling = type === 'gemini' ? wire.generationConfig : wire;
  assert.equal(sampling[outputField], 25600);
  assert.equal(sampling.temperature, 0.7);
  if (type === 'gemini') { assert.equal(wire.temperature, undefined); assert.equal(sampling.topP, 0.8); }
  else assert.equal(wire.max_output_tokens, undefined);
});

test('Luna max reaches official wire without an effort downgrade', async () => {
  let wire;
  const runtime = await createHostModelRuntime(config(), { fetch: async (_, init) => { wire = JSON.parse(init.body); return response(); } });
  const result = await complete(runtime, resolveTaskModel(runtime, 'helper'));
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  assert.equal(wire.reasoning.effort, 'max');
  assert.equal(wire.max_output_tokens, 4096);
});

test('unknown context uses SDK-supported 0 and strict opaque/native-media bounds are not asserted', async () => {
  const route = resolveTaskModel(await createHostModelRuntime(config()), 'main');
  assert.equal(route.model.contextWindow, 0);
  const cap = { mode: 'strict', contextKnown: true, contextWindow: 100000, source: 'operator-bound', input_counter: 'utf8_bytes_bound', tokens_per_utf8_byte_upper: 1, protocol_overhead_tokens: 0, output_semantics: 'includes_reasoning' };
  for (const input of [{ encrypted_content: 'opaque' }, { inlineData: { data: 'aGVsbG8=' } }, { thoughtSignature: 'opaque' }]) {
    assert.throws(() => checkRequestAdmission({ input, max_output_tokens: 20 }, cap, 'fixture'), /cannot establish/);
  }
});

test('model hook replacement fails before HTTP; credential/header changes isolate replay handles', async () => {
  let calls = 0;
  const runtime = await createHostModelRuntime(config(), { fetch: async () => { calls++; return response(); } });
  const result = await complete(runtime, resolveTaskModel(runtime, 'main'), { onPayload: (payload) => ({ ...payload, model: 'wrong-model' }) });
  assert.equal(result.stopReason, 'error'); assert.equal(calls, 0);
  const other = config(); other.api_providers[0].api_key = 'different-fixture';
  assert.notEqual(resolveTaskModel(runtime, 'main').routeFingerprint, resolveTaskModel(await createHostModelRuntime(other), 'main').routeFingerprint);
});

test('Google scoped fetch isolates two simultaneous routes, another provider and unscoped fetch', async () => {
  const original = globalThis.fetch;
  // The shim was installed by an earlier test; an unscoped data URL must still
  // reach Node's normal fetch rather than any route-specific mock or auth wrapper.
  const seen = [];
  const make = async (label, type) => {
    const cfg = config(); cfg.api_providers[0] = { ...cfg.api_providers[0], client_type: type, api_key: label,
      auth_type: 'header', auth_header_name: 'X-Route-Key', default_query: { route: label } };
    cfg.models[0].extra_params = {}; cfg.models[0].send_temperature = false;
    const runtime = await createHostModelRuntime(cfg, { fetch: async (url, init) => {
      await new Promise((resolve) => setTimeout(resolve, label === 'a' ? 10 : 1));
      seen.push({ label, url: String(url), headers: new Headers(init.headers), wire: JSON.parse(init.body) });
      return type === 'gemini' ? googleResponse(label) : response();
    } });
    return complete(runtime, resolveTaskModel(runtime, 'main'));
  };
  const results = await Promise.all([make('a', 'gemini'), make('b', 'gemini'), make('c', 'openai_responses'), globalThis.fetch('data:text/plain,unscoped').then((res) => res.text())]);
  assert.equal(results[3], 'unscoped');
  for (const result of results.slice(0, 3)) assert.equal(result.stopReason, 'stop', result.errorMessage);
  assert.equal(seen.length, 3);
  for (const entry of seen) {
    assert.equal(new URL(entry.url).searchParams.get('route'), entry.label);
    assert.equal(entry.headers.get('X-Route-Key'), entry.label);
    assert.equal(entry.headers.get('x-goog-api-key'), null);
    assert.equal(entry.headers.get('authorization'), null);
    assert.equal(entry.wire.generationConfig?.temperature, undefined);
  }
  assert.equal(globalThis.fetch, original);
});

test('Google cancellation and provider timeout reach the scoped transport', async () => {
  for (const useTimeout of [false, true]) {
    const cfg = config(); cfg.api_providers[0].client_type = 'gemini'; cfg.models[0].extra_params = {};
    if (useTimeout) cfg.api_providers[0].timeout = 0.02;
    let reached;
    const entered = new Promise((resolve) => { reached = resolve; });
    const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => {
      reached();
      return new Promise((_, reject) => {
        const abort = () => reject(init.signal.reason);
        if (init.signal.aborted) abort(); else init.signal.addEventListener('abort', abort, { once: true });
      });
    } });
    const controller = new AbortController();
    const pending = complete(runtime, resolveTaskModel(runtime, 'main'), { signal: controller.signal, maxRetries: 0 });
    await entered;
    if (!useTimeout) controller.abort();
    // Keep one ref'ed timer so Node's unref'ed AbortSignal.timeout can fire.
    const keepAlive = setTimeout(() => {}, 500);
    const result = await pending; clearTimeout(keepAlive);
    assert.equal(result.stopReason, 'aborted', result.errorMessage);
  }
});

test('provider transport forbids credential-bearing redirects for every auth scheme', async () => {
  const { createProviderFetch } = await import('../provider.mjs');
  for (const auth_type of ['bearer', 'header', 'query', 'none']) {
    let init;
    const fetcher = createProviderFetch({ auth_type, api_key: 'fixture-secret', auth_header_name: 'X-Key' }, {}, async (_, options) => { init = options; return new Response(''); });
    await fetcher('https://unit.invalid/v1', { redirect: 'follow' });
    assert.equal(init.redirect, 'error');
  }
});

test('Anthropic native replay leaves all old input blocks byte-identical without moving cache markers', async () => {
  const cfg = config(); cfg.api_providers[0].client_type = 'anthropic'; cfg.models[0].extra_params = {};
  const requests = [];
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { requests.push(JSON.parse(init.body)); return anthropicResponse(); } });
  const route = resolveTaskModel(runtime, 'main');
  const first = [{ role: 'user', content: 'one', timestamp: 1 }];
  const result = await runtime.completeSimple(route.model, { messages: first }, { cacheRetention: 'long' });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  const next = await runtime.completeSimple(route.model, { messages: [...first, result, { role: 'user', content: 'two', timestamp: 2 }] }, { cacheRetention: 'long' });
  assert.equal(next.stopReason, 'stop', next.errorMessage);
  assert.deepEqual(requests[1].messages.slice(0, requests[0].messages.length), requests[0].messages);
  assert.equal(JSON.stringify(requests).includes('cache_control'), false);
});

test('model fallback and preferred strict-schema fallback fail closed rather than silently rerouting or downgrading', async () => {
  await assert.rejects(createHostModelRuntime(config({ pi: { compat: { allowedFallbackModels: [{ provider: 'x', model: 'fallback' }] } } })), /fallback is not permitted/);
  let calls = 0;
  const cfg = config(); cfg.api_providers[0].client_type = 'anthropic'; cfg.models[0].extra_params = { pi: { compat: { supportsStrictTools: true } } };
  const runtime = await createHostModelRuntime(cfg, { fetch: async () => { calls++; return new Response(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: 'The compiled grammar is too large' } }), { status: 400, headers: { 'content-type': 'application/json' } }); } });
  const route = resolveTaskModel(runtime, 'main');
  const tool = { name: 'fixture', description: 'fixture', parameters: { type: 'object', properties: {} }, constrainedSampling: { type: 'json_schema', strict: 'prefer' } };
  const refused = await runtime.completeSimple(route.model, { messages: [{ role: 'system', content: '', toolsAdded: [tool] }, { role: 'user', content: 'test', timestamp: 1 }] });
  assert.equal(refused.stopReason, 'error'); assert.match(refused.errorMessage, /automatic schema fallback/);
  assert.equal(calls, 0);
  tool.constrainedSampling.strict = 'require';
  const result = await runtime.completeSimple(route.model, { messages: [{ role: 'system', content: '', toolsAdded: [tool] }, { role: 'user', content: 'test', timestamp: 1 }] }, { maxRetries: 0 });
  assert.equal(result.stopReason, 'error'); assert.equal(calls, 1);
});

for (const [type, responseFactory] of [['openai_responses', response], ['openai', () => new Response('data: {"id":"chat-1","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })], ['anthropic', anthropicResponse], ['gemini', googleResponse]]) {
  test(`request-level extraParams and toolChoice=none are honored by the official ${type} wire`, async () => {
    const cfg = config(); cfg.api_providers[0].client_type = type; cfg.models[0].extra_params = {};
    let wire;
    const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { wire = JSON.parse(init.body); return responseFactory(); } });
    const route = resolveTaskModel(runtime, 'main', { options: { toolChoice: 'none', extraParams: { top_p: 0.4 }, temperature: 0.2 } });
    assert.equal(route.requestOptions.toolChoice, 'none'); assert.equal(route.requestOptions.extraParams.top_p, 0.4);
    const result = await runtime.completeSimple(route.model, { messages: [
      { role: 'system', content: '', toolsAdded: [{ name: 'fixture', description: 'fixture', parameters: { type: 'object', properties: {} } }] },
      { role: 'user', content: 'test', timestamp: 1 },
    ] }, { onPayload: (payload) => ({ ...payload, tool_choice: 'auto' }) });
    assert.equal(result.stopReason, 'stop', result.errorMessage);
    if (type === 'gemini') { assert.equal(wire.toolConfig.functionCallingConfig.mode, 'NONE'); assert.equal(wire.generationConfig.topP, 0.4); }
    else { assert.deepEqual(wire.tool_choice, type === 'anthropic' ? { type: 'none' } : 'none'); assert.equal(wire.top_p, 0.4); }
    assert.notEqual(route.routeFingerprint, resolveTaskModel(runtime, 'main', { options: { toolChoice: 'auto', extraParams: { top_p: 0.4 }, temperature: 0.2 } }).routeFingerprint);
    assert.throws(() => resolveTaskModel(runtime, 'main', { options: { extraParams: { model: 'wrong' } } }), /cannot override/);
    assert.throws(() => resolveTaskModel(runtime, 'main', { options: { extraParams: { headers: { 'X-Wrong': 'wrong' } } } }), /cannot replace/);
    assert.throws(() => resolveTaskModel(runtime, 'main', { options: { toolChoice: 'required' } }), /auto or none/);
  });
}

test('simultaneous Google routes retain isolation when one request is canceled', async () => {
  const controllers = [new AbortController(), new AbortController()];
  const arrivals = []; const releases = [];
  let markReady;
  const ready = new Promise((resolve) => { markReady = resolve; });
  const runs = await Promise.all(['cancel', 'keep'].map(async (label, index) => {
    const cfg = config(); cfg.api_providers[0] = { ...cfg.api_providers[0], client_type: 'gemini', auth_type: 'header', auth_header_name: 'X-Route', api_key: label, default_query: { name: label } }; cfg.models[0].extra_params = {};
    const runtime = await createHostModelRuntime(cfg, { fetch: async (url, init) => {
      arrivals.push({ url: new URL(url), headers: new Headers(init.headers), signal: init.signal });
      if (arrivals.length === 2) markReady();
      return new Promise((resolve, reject) => {
        releases[index] = () => resolve(googleResponse(label));
        init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
      });
    } });
    return { pending: complete(runtime, resolveTaskModel(runtime, 'main'), { signal: controllers[index].signal, maxRetries: 0 }) };
  }));
  await ready; controllers[0].abort(); releases[1]();
  const [canceled, kept] = await Promise.all(runs.map(({ pending }) => pending));
  assert.equal(canceled.stopReason, 'aborted'); assert.equal(kept.stopReason, 'stop', kept.errorMessage);
  for (const arrival of arrivals) assert.equal(arrival.headers.get('X-Route'), arrival.url.searchParams.get('name'));
  assert.equal(arrivals.find((value) => value.headers.get('X-Route') === 'keep').signal.aborted, false);
});

test('Google onPayload hook unrelated fetch is never intercepted or given route credentials', async () => {
  const cfg = config(); cfg.api_providers[0].client_type = 'gemini'; cfg.models[0].extra_params = {};
  let requests = 0, hookValue;
  const runtime = await createHostModelRuntime(cfg, { fetch: async () => { requests++; return googleResponse(); } });
  const result = await complete(runtime, resolveTaskModel(runtime, 'main'), { onPayload: async (payload) => {
    hookValue = await globalThis.fetch('data:text/plain,unrelated').then((res) => res.text());
    return payload;
  } });
  assert.equal(result.stopReason, 'stop', result.errorMessage); assert.equal(hookValue, 'unrelated'); assert.equal(requests, 1);
});

test('official Gemini replays valid text/thinking/function-call signatures without rewriting the prior wire prefix', async () => {
  const cfg = config(); cfg.api_providers[0].client_type = 'gemini';
  cfg.models[0].model_identifier = 'gemini-3-fixture'; cfg.models[0].extra_params = { reasoning_effort: 'high' };
  const signatures = { thinking: Buffer.from('opaque thinking signature').toString('base64'), text: Buffer.from('opaque text signature').toString('base64'), tool: Buffer.from('opaque tool signature').toString('base64') };
  const requests = [];
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => {
    requests.push(JSON.parse(init.body));
    if (requests.length !== 1) return googleResponse('done');
    return new Response(`data: ${JSON.stringify({ candidates: [{ content: { role: 'model', parts: [
      { thought: true, text: 'private reasoning', thoughtSignature: signatures.thinking },
      { text: 'calling a tool', thoughtSignature: signatures.text },
      { functionCall: { id: 'google_call', name: 'fixture', args: { value: 1 } }, thoughtSignature: signatures.tool },
    ] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 } })}\n\n`, { headers: { 'content-type': 'text/event-stream' } });
  } });
  const route = resolveTaskModel(runtime, 'main');
  const prefix = [{ role: 'system', content: 'fixed', toolsAdded: [{ name: 'fixture', parameters: { type: 'object', properties: { value: { type: 'number' } } }, description: 'fixture' }] }, { role: 'user', content: 'first', timestamp: 1 }];
  const native = await runtime.completeSimple(route.model, { messages: prefix }, { reasoning: route.thinkingLevel });
  assert.equal(native.stopReason, 'toolUse', native.errorMessage);
  const call = native.content.find((block) => block.type === 'toolCall');
  const secondContext = [...prefix, native, { role: 'toolResult', toolCallId: call.id, toolName: call.name, content: [{ type: 'text', text: 'result' }], isError: false, timestamp: 2 }];
  const answer = await runtime.completeSimple(route.model, { messages: secondContext }, { reasoning: route.thinkingLevel });
  assert.equal(answer.stopReason, 'stop', answer.errorMessage);
  const replayed = requests[1].contents.find((entry) => entry.role === 'model').parts;
  assert.deepEqual(replayed.map((part) => part.thoughtSignature), [signatures.thinking, signatures.text, signatures.tool]);
  assert.equal(replayed[0].thought, true); assert.equal(replayed[2].functionCall.id, 'google_call');
  const third = await runtime.completeSimple(route.model, { messages: [...secondContext, answer, { role: 'user', content: 'next', timestamp: 3 }] }, { reasoning: route.thinkingLevel });
  assert.equal(third.stopReason, 'stop', third.errorMessage);
  assert.deepEqual(requests[2].contents.slice(0, requests[1].contents.length), requests[1].contents);
});

test('official Anthropic replays signed and redacted thinking plus tool results with identical native prefix', async () => {
  const cfg = config(); cfg.api_providers[0].client_type = 'anthropic';
  const requests = [];
  const signedResponse = () => {
    const events = [
      { type: 'message_start', message: { id: 'msg_native', role: 'assistant', model: 'exact-terra-route', content: [], usage: { input_tokens: 4, output_tokens: 0 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'private reasoning' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'opaque-anthropic-signature' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'content_block_start', index: 1, content_block: { type: 'redacted_thinking', data: 'opaque-redacted-thinking' } },
      { type: 'content_block_stop', index: 1 },
      { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 'anthropic_call', name: 'fixture', input: {} } },
      { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"value":1}' } },
      { type: 'content_block_stop', index: 2 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 5 } }, { type: 'message_stop' },
    ];
    return new Response(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });
  };
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { requests.push(JSON.parse(init.body)); return requests.length === 1 ? signedResponse() : anthropicResponse(); } });
  const route = resolveTaskModel(runtime, 'main');
  const prefix = [{ role: 'user', content: 'first', timestamp: 1 }];
  const native = await runtime.completeSimple(route.model, { messages: prefix }, { reasoning: route.thinkingLevel });
  assert.equal(native.stopReason, 'toolUse', native.errorMessage);
  const call = native.content.find((block) => block.type === 'toolCall');
  const secondContext = [...prefix, native, { role: 'toolResult', toolCallId: call.id, toolName: call.name, content: [{ type: 'text', text: 'result' }], isError: false, timestamp: 2 }];
  const answer = await runtime.completeSimple(route.model, { messages: secondContext }, { reasoning: route.thinkingLevel });
  assert.equal(answer.stopReason, 'stop', answer.errorMessage);
  const replayed = requests[1].messages[1].content;
  assert.deepEqual(replayed[0], { type: 'thinking', thinking: 'private reasoning', signature: 'opaque-anthropic-signature' });
  assert.deepEqual(replayed[1], { type: 'redacted_thinking', data: 'opaque-redacted-thinking' });
  assert.equal(requests[1].messages[2].content[0].tool_use_id, call.id);
  const third = await runtime.completeSimple(route.model, { messages: [...secondContext, answer, { role: 'user', content: 'next', timestamp: 3 }] }, { reasoning: route.thinkingLevel });
  assert.equal(third.stopReason, 'stop', third.errorMessage);
  assert.deepEqual(requests[2].messages.slice(0, requests[1].messages.length), requests[1].messages);
});

test('no output budget is invented for a model without model/task configuration', async () => {
  const cfg = config(); delete cfg.models[0].max_tokens; cfg.models[0].extra_params = {}; delete cfg.model_task_config.main.max_tokens;
  const runtime = await createHostModelRuntime(cfg);
  assert.throws(() => resolveTaskModel(runtime, 'main'), /positive integer/);
  assert.equal(resolveTaskModel(runtime, 'main', { options: { maxTokens: 777 } }).maxTokens, 777);
});

// Migration coverage for the retired Python SDK admission fixtures. The Node
// runtime supports declared whole-JSON byte bounds, not Python counter plugins.
const admissionConfig = (overrides = {}) => {
  const cfg = config(); delete cfg.api_providers[0].default_headers;
  cfg.models[0].max_tokens = 100; cfg.models[0].extra_params = {};
  cfg.models[0].request_admission = { mode: 'strict', capability_source: 'offline exact route declaration',
    context_window: 10000, safety_margin_tokens: 7, output_semantics: 'includes_reasoning',
    input_counter: 'utf8_bytes_bound', tokens_per_utf8_byte_upper: 1, protocol_overhead_tokens: 3, ...overrides };
  return cfg;
};
const smallContext = { messages: [{ role: 'user', content: '汉🙂', timestamp: 1 }] };

test('final serialized schema/tools bytes are counted, while schema property names are not mistaken for media or references', async () => {
  let sent;
  const runtime = await createHostModelRuntime(admissionConfig(), { fetch: async (_, init) => { sent = init.body; return response(); } });
  const route = resolveTaskModel(runtime, 'main');
  const schema = { type: 'object', properties: Object.fromEntries(['image_url', 'data', 'signature', 'file_id', 'prompt'].map((key) => [key, { type: 'string' }])), description: 'schema text'.repeat(30) };
  const context = { messages: [{ role: 'system', content: '', toolsAdded: [{ name: 'inspect', description: 'inspect', parameters: schema }] }, ...smallContext.messages] };
  const result = await runtime.completeSimple(route.model, context, { onPayload: (payload) => ({ ...payload,
    text: { format: { type: 'json_schema', name: 'output', schema } }, prompt_cache_key: 'cached-input-is-still-input' }) });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  const wire = JSON.parse(sent);
  assert.equal(wire.max_output_tokens, 100); assert.equal(wire.tools[0].parameters.properties.image_url.type, 'string');
  const measured = checkRequestAdmission(wire, route.capability, route.model.api, sent);
  assert.equal(measured.boundedInput, Buffer.byteLength(sent) + 3); assert.equal(measured.verified, true);
});

test('post-conversion body, tool and structured-output expansion is refused before any HTTP send without lowering reserve', async () => {
  for (const kind of ['body', 'tools', 'schema', 'output']) {
    const cfg = admissionConfig({ context_window: 700 });
    if (kind === 'output') cfg.models[0].max_tokens = 1000;
    let calls = 0;
    const runtime = await createHostModelRuntime(cfg, { fetch: async () => { calls++; return response(); } });
    const route = resolveTaskModel(runtime, 'main');
    const hook = (payload) => kind === 'body' ? { ...payload, metadata: { expanded: 'x'.repeat(3000) } }
      : kind === 'tools' ? { ...payload, tools: [{ type: 'function', name: 'inspect', description: 'x'.repeat(3000), parameters: { type: 'object', properties: {} } }] }
      : kind === 'schema' ? { ...payload, text: { format: { type: 'json_schema', name: 'output', schema: { type: 'object', description: 'x'.repeat(3000) } } } } : payload;
    const result = await runtime.completeSimple(route.model, smallContext, { onPayload: hook });
    assert.equal(result.stopReason, 'error'); assert.match(result.errorMessage, /exceeds/); assert.equal(calls, 0);
    assert.equal(route.maxTokens, kind === 'output' ? 1000 : 100);
  }
});

test('server-managed state, built-in tools and unknown envelopes remain unknown in strict mode but audit stays usable', async () => {
  const additions = [
    { previous_response_id: 'remote-state' }, { conversation: 'remote-conversation' },
    { prompt: { id: 'server-template', variables: { value: 'small-reference' } } },
    { unknown_provider_extension: { id: 'hidden-input' } }, { tools: [{ type: 'web_search' }] },
    { input: [{ type: 'reasoning', encrypted_content: 'opaque' }] },
    { input: [{ role: 'user', content: [{ type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=' }] }] },
  ];
  for (const mode of ['strict', 'audit']) {
    let calls = 0;
    const runtime = await createHostModelRuntime(admissionConfig({ mode }), { fetch: async () => { calls++; return response(); } });
    const route = resolveTaskModel(runtime, 'main');
    for (const extra of additions) {
      const result = await runtime.completeSimple(route.model, smallContext, { onPayload: (payload) => ({ ...payload, ...extra }) });
      assert.equal(result.stopReason, mode === 'strict' ? 'error' : 'stop', result.errorMessage);
      if (mode === 'strict') assert.match(result.errorMessage, /cannot establish/);
    }
    assert.equal(calls, mode === 'strict' ? 0 : additions.length);
  }
});

test('unsupported counters and incomplete proof configuration refuse locally; transport hashes do not invent an exact counter', async () => {
  for (const overrides of [
    { input_counter: 'unknown' }, { input_counter: 'retired-python-exact-counter' },
    { tokens_per_utf8_byte_upper: null }, { protocol_overhead_tokens: null },
    { protocol_overhead_tokens: -1 }, { context_window: null },
    { capability_source: '' }, { output_semantics: 'unknown' }, { output_semantics: 'separate_reasoning', reasoning_tokens_upper: 70 },
    { transport_contract_fingerprint: 'a'.repeat(64) },
  ]) {
    const cfg = admissionConfig(overrides);
    if (overrides.transport_contract_fingerprint) cfg.api_providers[0].default_headers = { 'X-Upstream-Route': 'declared-but-not-counted-exactly' };
    let calls = 0;
    const runtime = await createHostModelRuntime(cfg, { fetch: async () => { calls++; return response(); } });
    const result = await runtime.completeSimple(resolveTaskModel(runtime, 'main').model, smallContext);
    assert.equal(result.stopReason, 'error'); assert.match(result.errorMessage, /cannot establish/); assert.equal(calls, 0);
  }
});

test('known input-only or output-only overflow is not hidden by unknown complementary accounting', () => {
  const cap = { mode: 'audit', contextKnown: true, contextWindow: 200, source: 'declared', safety_margin_tokens: 0,
    input_counter: 'utf8_bytes_bound', tokens_per_utf8_byte_upper: 1, protocol_overhead_tokens: 0, output_semantics: 'unknown' };
  assert.throws(() => checkRequestAdmission({ input: 'x'.repeat(500) }, cap, 'openai-responses'), /input bound alone exceeds/);
  assert.throws(() => checkRequestAdmission({ max_output_tokens: 500 }, { ...cap, input_counter: 'unknown' }, 'openai-responses'), /Output allowance alone exceeds/);
});

test('provider errors never trigger unchanged retries, compatibility rewrites or another configured model', async () => {
  for (const status of [400, 429, 503]) {
    const cfg = admissionConfig(); cfg.model_task_config.main.model_list = ['main', 'helper']; cfg.api_providers[0].max_retry = 3;
    const wires = [];
    const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => {
      wires.push(JSON.parse(init.body)); return new Response(JSON.stringify({ error: { message: "Use 'max_completion_tokens' instead.", type: 'invalid_request_error' } }), { status, headers: { 'content-type': 'application/json' } });
    } });
    const route = resolveTaskModel(runtime, 'main');
    const result = await runtime.completeSimple(route.model, smallContext, { maxRetries: 9 });
    assert.equal(result.stopReason, 'error'); assert.equal(wires.length, 1);
    assert.equal(wires[0].model, 'exact-terra-route'); assert.equal(wires[0].max_output_tokens, 100);
    assert.equal(wires[0].max_completion_tokens, undefined);
  }
});

test('runtime route and request snapshots cannot be mutated while an async payload hook is pending', async () => {
  const cfg = admissionConfig(); const options = { extraParams: { metadata: { marker: 'original' } } };
  let wire, auth;
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { wire = JSON.parse(init.body); auth = new Headers(init.headers).get('authorization'); return response(); } });
  const route = resolveTaskModel(runtime, 'main', { options });
  const result = await runtime.completeSimple(route.model, smallContext, { onPayload: async (payload) => {
    cfg.api_providers[0].api_key = 'late-mutated-secret'; cfg.models[0].request_admission.context_window = 1;
    options.extraParams.metadata.marker = 'late mutation';
    await Promise.resolve(); return payload;
  } });
  assert.equal(result.stopReason, 'stop', result.errorMessage); assert.equal(auth, 'Bearer test-secret');
  assert.equal(wire.metadata.marker, 'original'); assert.equal(route.capability.contextWindow, 10000);
});

test('simultaneous strict admissions are isolated per selected model and do not impose a global strict scope', async () => {
  const cfg = admissionConfig({ context_window: 700 }); cfg.models[1].max_tokens = 100; cfg.models[1].extra_params = {};
  cfg.models[1].request_admission = { ...cfg.models[0].request_admission, mode: 'audit', context_window: null, input_counter: 'unknown' };
  const wires = [];
  const runtime = await createHostModelRuntime(cfg, { fetch: async (_, init) => { wires.push(JSON.parse(init.body)); return response(); } });
  const context = { messages: [{ role: 'user', content: 'x'.repeat(1000), timestamp: 1 }] };
  const [small, audit] = await Promise.all(['main', 'helper'].map((task) => runtime.completeSimple(resolveTaskModel(runtime, task).model, context)));
  assert.equal(small.stopReason, 'error'); assert.equal(audit.stopReason, 'stop', audit.errorMessage);
  assert.equal(wires.length, 1); assert.equal(wires[0].model, 'exact-luna-route');
});
