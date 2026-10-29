import assert from 'node:assert/strict';
import test from 'node:test';
import { createHostModelRuntime } from '../provider.mjs';
import { runAuxiliaryOperation } from '../auxiliary.mjs';

const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } });
function config({ type = 'openai', modelType = 'embedding', extra = {}, base = 'https://unit.invalid/v1', provider = {} } = {}) {
  return { api_providers: [{ name: 'aux', client_type: type, base_url: base, api_key: 'fixture-secret', ...provider }],
    models: [{ name: 'model', model_identifier: 'exact-model', model_type: modelType, api_provider: 'aux', extra_params: extra }],
    model_task_config: { job: { model_list: ['model'], max_tokens: 512 } } };
}
const run = (runtime, operation, input, options = {}, extra = {}) => runAuxiliaryOperation(runtime, operation, { taskName: 'job', input, options }, extra);
const image = { data: 'aW1hZ2U=', mimeType: 'image/png', preprocessVersion: 'v1' };

test('OpenAI text embeddings retain effective dimensions, usage, auth and canonical request identity', async () => {
  let wire, headers, url;
  const runtime = await createHostModelRuntime(config({ extra: { body: { dimensions: 3 }, headers: { 'X-Route': 'one' }, query: { version: '1' } },
    provider: { auth_type: 'query', auth_query_name: 'access_key' } }), { fetch: async (target, init) => {
      url = new URL(target); headers = new Headers(init.headers); wire = JSON.parse(init.body);
      return json({ data: [{ embedding: [1, 2] }], usage: { total_tokens: 5 } });
    } });
  const result = await run(runtime, 'embed_text', 'one', { dimensions: 2 });
  assert.equal(wire.model, 'exact-model'); assert.equal(wire.dimensions, 2); assert.equal(wire.input, 'one');
  assert.equal(headers.get('authorization'), null); assert.equal(headers.get('X-Route'), 'one');
  assert.equal(url.searchParams.get('access_key'), 'fixture-secret'); assert.equal(url.searchParams.get('version'), '1');
  assert.deepEqual(result.embedding, [1, 2]); assert.deepEqual(result.usage, { total_tokens: 5 });
  assert.equal(result.route.nativePiOperation, false);
  const next = await run(runtime, 'embed_text', 'different source', { dimensions: 2 });
  assert.equal(next.requestProtocolHash, result.requestProtocolHash, 'source text is not protocol identity');
  for (const extraParams of [{ headers: { 'X-Route': 'two' } }, { query: { version: 'two' } }, { pi: { imageEmbeddingProtocol: 'wrong' } }]) {
    await assert.rejects(run(runtime, 'embed_text', 'one', { dimensions: 2, extraParams }), /cannot replace configured/);
  }
  const duplicate = await run(runtime, 'embed_text', 'one', { dimensions: 2, extraParams: { headers: { 'X-Route': 'one' }, query: { version: '1' } } });
  assert.equal(duplicate.requestProtocolHash, result.requestProtocolHash);
  const other = await createHostModelRuntime(config({ extra: { body: { dimensions: 2 }, headers: { 'X-Route': 'two' }, query: { version: '1' } }, provider: { auth_type: 'query', auth_query_name: 'access_key' } }), { fetch: async () => json({ data: [{ embedding: [1, 2] }] }) });
  assert.notEqual((await run(other, 'embed_text', 'one')).requestProtocolHash, result.requestProtocolHash);
});

test('explicit extraParamsReplace never silently reintroduces model dimensions; actual output dimensions fingerprint', async () => {
  let dimensions = 2, wire;
  const runtime = await createHostModelRuntime(config({ extra: { dimensions: 3 } }), { fetch: async (_, init) => {
    wire = JSON.parse(init.body); return json({ data: [{ embedding: Array(dimensions).fill(1) }] });
  } });
  const a = await run(runtime, 'embed_text', 'text', { extraParamsReplace: true, extraParams: {} });
  assert.equal(wire.dimensions, undefined);
  dimensions = 4;
  const b = await run(runtime, 'embed_text', 'text', { extraParamsReplace: true, extraParams: {} });
  assert.notEqual(a.requestProtocolHash, b.requestProtocolHash);
  await assert.rejects(run(runtime, 'embed_text', 'text', { dimensions: 3 }), /dimension does not match/);
});

test('Gemini embedContent keeps outputDimensionality/taskType and provider query auth', async () => {
  let wire, headers, url;
  const runtime = await createHostModelRuntime(config({ type: 'gemini', extra: { output_dimensionality: 2, task_type: 'RETRIEVAL_DOCUMENT' }, provider: { auth_type: 'query' } }), { fetch: async (target, init) => {
    url = new URL(target); headers = new Headers(init.headers); wire = JSON.parse(init.body); return json({ embedding: { values: [1, 2] }, usageMetadata: { totalTokenCount: 9 } });
  } });
  const result = await run(runtime, 'embed_text', 'document');
  assert.ok(url.pathname.endsWith('/models/exact-model:embedContent'));
  assert.equal(headers.get('x-goog-api-key'), null); assert.equal(url.searchParams.get('api_key'), 'fixture-secret');
  assert.equal(wire.outputDimensionality, 2); assert.equal(wire.taskType, 'RETRIEVAL_DOCUMENT');
  assert.deepEqual(result.usage, { totalTokenCount: 9 });
});

for (const [protocol, response, assertBody] of [
  ['qwen3vl-retrieval-v1', { data: [{ embedding: [1, 2] }] }, (body) => assert.equal(body.messages[0].content[0].image_url.url, 'data:image/png;base64,aW1hZ2U=')],
  ['dashscope', { output: { embeddings: [{ embedding: [1, 2] }] } }, (body) => { assert.equal(body.input.contents[0].image, 'data:image/png;base64,aW1hZ2U='); assert.equal(body.parameters.dimension, 2); }],
  ['ark', { data: { embedding: [1, 2] } }, (body) => assert.equal(body.input[0].type, 'image_url')],
  ['siliconflow', { data: [{ embedding: [1, 2] }] }, (body) => assert.equal(body.input.image, 'data:image/png;base64,aW1hZ2U=')],
]) test(`image embedding ${protocol} preserves explicit protocol and preprocessing identity`, async () => {
  let wire;
  const runtime = await createHostModelRuntime(config({ modelType: 'image_embedding', extra: { dimensions: 2, pi: { imageEmbeddingProtocol: protocol } } }), { fetch: async (_, init) => { wire = JSON.parse(init.body); return json(response); } });
  const a = await run(runtime, 'embed_image', image); assertBody(wire); assert.equal(wire.model, 'exact-model');
  const b = await run(runtime, 'embed_image', { ...image, preprocessVersion: 'v2' });
  assert.notEqual(a.requestProtocolHash, b.requestProtocolHash);
});

test('explicit image template retains placeholders but cannot replace the configured model', async () => {
  let wire;
  const runtime = await createHostModelRuntime(config({ modelType: 'image_embedding', extra: {
    image_embedding_body: { model: 'untrusted-template', metadata: { mime: '{mime_type}' } }, image_embedding_input: [{ url: '{data_uri}' }],
  } }), { fetch: async (_, init) => { wire = JSON.parse(init.body); return json({ data: [{ embedding: [1] }] }); } });
  await run(runtime, 'embed_image', image);
  assert.equal(wire.model, 'exact-model'); assert.equal(wire.metadata.mime, 'image/png');
  assert.equal(wire.input[0].url, 'data:image/png;base64,aW1hZ2U=');
});

test('ASR OpenAI multipart and Gemini native payload retain text/usage and exact task budget', async () => {
  for (const type of ['openai', 'gemini']) {
    let body;
    const runtime = await createHostModelRuntime(config({ type, modelType: 'llm', extra: { temperature: 0.2 } }), { fetch: async (_, init) => {
      body = init.body instanceof FormData ? init.body : JSON.parse(init.body);
      return json(type === 'openai' ? { text: 'transcript', usage: { input_tokens: 3 } } : { candidates: [{ content: { parts: [{ text: 'transcript' }] } }], usageMetadata: { totalTokenCount: 3 } });
    } });
    const result = await run(runtime, 'transcribe', { data: 'YXVkaW8=', mimeType: 'audio/wav' });
    assert.equal(result.text, 'transcript'); assert.ok(result.usage);
    if (type === 'openai') { assert.equal(body.get('model'), 'exact-model'); assert.equal(body.get('file').type, 'audio/wav'); }
    else { assert.equal(body.generationConfig.maxOutputTokens, 512); assert.equal(body.generationConfig.temperature, 0.2); }
    await assert.rejects(run(runtime, 'transcribe', { data: 'YXVkaW8=', mimeType: 'audio/wav' }, { extraParams: { model: 'wrong' } }), /cannot override/);
  }
});

test('auxiliary failures are explicit; invalid data, HTTP errors and cancellation do not reroute', async () => {
  let calls = 0;
  const runtime = await createHostModelRuntime(config(), { fetch: async () => { calls++; return json({ error: 'do not expose provider body' }, 401); } });
  await assert.rejects(run(runtime, 'embed_text', 'text'), { code: 'MODEL_HTTP_ERROR', status: 401 });
  assert.equal(calls, 1);
  await assert.rejects(run(runtime, 'embed_text', 'text', { dimensions: -1 }), /positive integer/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(run(runtime, 'embed_text', 'text', {}, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(calls, 1);
  const invalid = await createHostModelRuntime(config(), { fetch: async () => json({ data: [{ embedding: [1, 'bad'] }] }) });
  await assert.rejects(run(invalid, 'embed_text', 'text'), /finite vector/);
});

test('auxiliary mid-flight cancellation reaches transport', async () => {
  const controller = new AbortController(); let entered;
  const ready = new Promise((resolve) => { entered = resolve; });
  const runtime = await createHostModelRuntime(config(), { fetch: async (_, init) => { entered(); return new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true })); } });
  const pending = run(runtime, 'embed_text', 'text', {}, { signal: controller.signal });
  await ready; controller.abort(); await assert.rejects(pending, { name: 'AbortError' });
});

test('explicit strict non-chat operations refuse unknown operation bounds before transport; dimensions are not chat output budgets', async () => {
  for (const [operation, modelType, input, extra] of [
    ['embed_text', 'embedding', 'text', {}], ['embed_image', 'image_embedding', image, { pi: { imageEmbeddingProtocol: 'qwen3vl-retrieval-v1' } }],
    ['transcribe', 'llm', { data: 'YXVkaW8=', mimeType: 'audio/wav' }, {}], ['generate_image', 'llm', 'draw', { pi: { operation: 'image' } }],
  ]) {
    const cfg = config({ modelType, extra });
    cfg.models[0].request_admission = { mode: 'strict', capability_source: 'declared chat contract is not an operation-specific proof', context_window: 10000, max_output_tokens: 1000, output_semantics: 'includes_reasoning', input_counter: 'utf8_bytes_bound', tokens_per_utf8_byte_upper: 1, protocol_overhead_tokens: 0 };
    let calls = 0;
    const runtime = await createHostModelRuntime(cfg, { fetch: async () => { calls++; return json({}); } });
    await assert.rejects(run(runtime, operation, input), { code: 'CONTEXT_CAPABILITY_UNKNOWN' });
    assert.equal(calls, 0);
    if (operation === 'generate_image') {
      const native = runtime.getModelOfType('image', 'maibot:aux:model', 'exact-model');
      const result = await runtime.generateImages(native, { input: [{ type: 'text', text: 'draw' }] });
      assert.equal(result.stopReason, 'error'); assert.match(result.errorMessage, /cannot establish/); assert.equal(calls, 0);
    }
  }
});
