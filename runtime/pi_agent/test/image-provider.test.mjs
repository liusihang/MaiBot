import assert from 'node:assert/strict';
import test from 'node:test';
import { createHostModelRuntime } from '../provider.mjs';
import { runAuxiliaryOperation } from '../auxiliary.mjs';
import { makeImageProvider } from '../image-provider.mjs';
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'x-fixture': 'true' } });
const provider = { base_url: 'https://provider.invalid/v1', api_key: 'fixture-secret' };
const model = { id: 'exact-image', api: 'maibot-openai-images', provider: 'fixture', cost: { input: 1, output: 2, cacheRead: 0.5, cacheWrite: 0 }, output: ['text', 'image'] };
const context = { input: [{ type: 'text', text: 'draw a flower' }] };
const picture = { b64_json: 'aW1hZ2U=', revised_prompt: 'flower' };

test('image generation follows native Pi response hook and typed usage contracts', async () => {
  let body, observed;
  const adapter = makeImageProvider(provider, { body: { size: '1024x1024' } }, async (_, init) => { body = JSON.parse(init.body); return json({ id: 'image-1', data: [picture], usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 } }); });
  const result = await adapter.generateImages(model, context, { metadata: { quality: 'high' }, onResponse: (response) => { observed = response; } });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  assert.equal(body.model, 'exact-image'); assert.equal(body.size, '1024x1024'); assert.equal(body.quality, 'high');
  assert.deepEqual(observed, { status: 200, headers: { 'content-type': 'application/json', 'x-fixture': 'true' } });
  assert.equal(result.usage.totalTokens, 30); assert.equal(result.usage.input, 10); assert.equal(result.usage.output, 20);
  assert.equal(result.responseId, 'image-1'); assert.equal(result.output[0].mimeType, 'image/png'); assert.equal(result.output[1].text, 'flower');
});

test('URL-only image output downloads without provider credentials across redirects', async () => {
  const downloads = [];
  const adapter = makeImageProvider(provider, { body: {} }, async () => json({ data: [{ url: 'https://93.184.216.34/image?signature=public-output-token' }] }), async (url, init) => {
    downloads.push({ url: String(url), init });
    if (downloads.length === 1) return new Response(null, { status: 302, headers: { location: 'https://93.184.216.35/final.png' } });
    return new Response(Buffer.from('image bytes'), { headers: { 'content-type': 'image/png' } });
  });
  const result = await adapter.generateImages(model, context);
  assert.equal(result.stopReason, 'stop', result.errorMessage); assert.equal(downloads.length, 2);
  for (const { init, url } of downloads) {
    assert.equal(new Headers(init.headers).get('authorization'), null);
    assert.equal(new Headers(init.headers).get('x-api-key'), null);
    assert.equal(init.credentials, 'omit'); assert.equal(init.redirect, 'manual');
    assert.equal(url.includes('fixture-secret'), false);
  }
  assert.equal(Buffer.from(result.output[0].data, 'base64').toString(), 'image bytes');
});

test('unsafe image URLs and unsafe redirects are refused without downloading them', async () => {
  for (const url of ['http://93.184.216.34/image', 'https://127.0.0.1/image', 'https://user:secret@93.184.216.34/image', 'https://[::1]/image', 'https://localhost/image']) {
    let downloads = 0;
    const result = await makeImageProvider(provider, { body: {} }, async () => json({ data: [{ url }] }), async () => { downloads++; return new Response(''); }).generateImages(model, context);
    assert.equal(result.stopReason, 'error'); assert.match(result.errorMessage, /public HTTPS/); assert.equal(downloads, 0);
  }
  let downloads = 0;
  const result = await makeImageProvider(provider, { body: {} }, async () => json({ data: [{ url: 'https://93.184.216.34/image' }] }), async () => {
    downloads++; return new Response(null, { status: 307, headers: { location: 'https://169.254.169.254/metadata' } });
  }).generateImages(model, context);
  assert.equal(result.stopReason, 'error'); assert.equal(downloads, 1);
});

test('image edit FormData identity cannot be overwritten by metadata or payload hooks', async () => {
  const input = { input: [...context.input, { type: 'image', mimeType: 'image/png', data: 'aW1hZ2U=' }] };
  let calls = 0, body;
  const adapter = makeImageProvider(provider, { body: {} }, async (_, init) => { calls++; body = init.body; return json({ data: [picture] }); });
  const result = await adapter.generateImages(model, input, { metadata: { quality: 'high' } });
  assert.equal(result.stopReason, 'stop', result.errorMessage); assert.equal(body.get('model'), 'exact-image');
  assert.equal(body.get('image').type, 'image/png'); assert.equal(body.get('quality'), 'high');
  const bad = await adapter.generateImages(model, input, { metadata: { model: 'wrong' } });
  assert.equal(bad.stopReason, 'error'); assert.equal(calls, 1);
  const hook = await adapter.generateImages(model, input, { onPayload: (payload) => { payload.set('model', 'wrong'); } });
  assert.equal(hook.stopReason, 'error'); assert.equal(calls, 1);
});

test('Google image adapter retains image inputs, generation options, text and usage', async () => {
  let body, headers;
  const adapter = makeImageProvider(provider, { body: { imageConfig: { aspectRatio: '1:1' } } }, async (_, init) => {
    body = JSON.parse(init.body); headers = new Headers(init.headers);
    return json({ candidates: [{ content: { parts: [{ text: 'done' }, { inlineData: { mimeType: 'image/webp', data: 'aW1hZ2U=' } }] } }], usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 5, totalTokenCount: 8 } });
  });
  const result = await adapter.generateImages({ ...model, api: 'maibot-google-images' }, { input: [...context.input, { type: 'image', mimeType: 'image/png', data: 'aW1hZ2U=' }] });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  assert.equal(headers.get('authorization'), null); assert.equal(headers.get('x-goog-api-key'), 'fixture-secret');
  assert.equal(body.generationConfig.imageConfig.aspectRatio, '1:1'); assert.equal(body.contents[0].parts[1].inlineData.mimeType, 'image/png');
  assert.equal(result.usage.totalTokens, 8); assert.equal(result.output[1].mimeType, 'image/webp');
});

test('image native ModelRuntime registration retains selected route, caller options and usage', async () => {
  let wire, headers;
  const config = { api_providers: [{ name: 'image', client_type: 'openai', ...provider }], models: [{ name: 'image-model', model_identifier: 'exact-image', api_provider: 'image', extra_params: { pi: { operation: 'image' }, headers: { 'X-Image': 'yes' }, size: '1024x1024' } }], model_task_config: { image: { model_list: ['image-model'] } } };
  const runtime = await createHostModelRuntime(config, { fetch: async (_, init) => { wire = JSON.parse(init.body); headers = new Headers(init.headers); return json({ data: [picture], usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 } }); } });
  const result = await runAuxiliaryOperation(runtime, 'generate_image', { taskName: 'image', input: 'flower', options: { quality: 'high', extraParams: { headers: { 'X-Image': 'yes' } } } });
  assert.equal(result.route.nativePiOperation, true); assert.equal(result.route.modelIdentifier, 'exact-image');
  assert.equal(wire.quality, 'high'); assert.equal(wire.size, '1024x1024'); assert.equal(wire.extraParams, undefined);
  assert.equal(headers.get('X-Image'), 'yes'); assert.equal(result.usage.totalTokens, 3);
});

test('image invalid output, HTTP failure, oversize downloads and cancellation are explicit', async () => {
  for (const raw of [{ data: [] }, { data: [picture], error: 'irrelevant' }]) {
    const status = raw.data.length ? 403 : 200;
    const result = await makeImageProvider(provider, { body: {} }, async () => json(raw, status)).generateImages(model, context);
    assert.equal(result.stopReason, 'error');
  }
  const tooLarge = await makeImageProvider(provider, { body: {} }, async () => json({ data: [{ url: 'https://93.184.216.34/image' }] }), async () => new Response('a', { headers: { 'content-type': 'image/png', 'content-length': String(33 * 1024 * 1024) } })).generateImages(model, context);
  assert.equal(tooLarge.stopReason, 'error'); assert.match(tooLarge.errorMessage, /size limit/);
  const controller = new AbortController(); controller.abort(); let calls = 0;
  const canceled = await makeImageProvider(provider, { body: {} }, async () => { calls++; return json({ data: [picture] }); }).generateImages(model, context, { signal: controller.signal });
  assert.equal(canceled.stopReason, 'aborted'); assert.equal(calls, 0);
});

test('public HTTPS transport pins vetted DNS addresses and handles empty-body statuses without uncaught errors', async () => {
  const { EventEmitter } = await import('node:events');
  const { Readable } = await import('node:stream');
  const { publicImageFetch } = await import('../image-provider.mjs');
  for (const statusCode of [200, 204, 205, 304]) {
    let pinned, stream;
    const response = await publicImageFetch('https://cdn.invalid/image', {
      lookup: async () => [{ address: '93.184.216.34', family: 4 }],
      request: (_, options, callback) => {
        options.lookup('cdn.invalid', { all: true }, (_, addresses) => { pinned = addresses; });
        const req = new EventEmitter();
        req.end = () => { stream = Readable.from([Buffer.from('image')]); stream.statusCode = statusCode; stream.headers = { 'content-type': 'image/png' }; callback(stream); };
        return req;
      },
    });
    assert.equal(response.status, statusCode); assert.deepEqual(pinned, [{ address: '93.184.216.34', family: 4 }]);
    if (statusCode === 200) assert.equal(await response.text(), 'image');
    else { assert.equal(response.body, null); assert.equal(stream.destroyed, true); }
  }
  for (const address of ['127.0.0.1', '10.0.0.1', '2002:7f00:1::', '2001:db8::1', '2001:20::1']) {
    await assert.rejects(publicImageFetch('https://cdn.invalid/image', { lookup: async () => [{ address, family: address.includes(':') ? 6 : 4 }], request: () => { throw new Error('must not open socket'); } }), /non-public/);
  }
});

test('official OpenRouter image provider keeps native data images and safely recovers URL-only images', async () => {
  for (const imageUrl of ['data:image/png;base64,aW1hZ2U=', 'https://93.184.216.34/generated.png']) {
    const downloads = []; let wire;
    const adapter = makeImageProvider(provider, { body: { image_config: { aspect_ratio: '1:1' } } }, async (_, init) => {
      wire = JSON.parse(init.body);
      return json({ id: 'router-image', choices: [{ message: { content: 'generated', images: [{ image_url: { url: imageUrl } }] } }], usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 } });
    }, async (url, init) => { downloads.push({ url: String(url), init }); return new Response('image bytes', { headers: { 'content-type': 'image/png' } }); });
    const result = await adapter.generateImages({ ...model, api: 'openrouter-images', baseUrl: provider.base_url }, context, { apiKey: 'fixture-secret' });
    assert.equal(result.stopReason, 'stop', result.errorMessage); assert.equal(result.output.filter((item) => item.type === 'image').length, 1);
    assert.equal(result.responseId, 'router-image'); assert.equal(result.usage.totalTokens, 7); assert.equal(wire.model, 'exact-image');
    assert.equal(downloads.length, imageUrl.startsWith('data:') ? 0 : 1);
    for (const { init } of downloads) { assert.equal(new Headers(init.headers).get('authorization'), null); assert.equal(init.credentials, 'omit'); }
  }
});

test('official OpenRouter empty or text-only image results fail explicitly; HTTP errors never retry', async () => {
  for (const [message, status] of [[{}, 200], [{ content: 'Generated image but no image attached' }, 200], [{}, 503]]) {
    let calls = 0;
    const adapter = makeImageProvider(provider, { body: {} }, async () => { calls++; return json({ choices: [{ message }] }, status); });
    const result = await adapter.generateImages({ ...model, api: 'openrouter-images', baseUrl: provider.base_url }, context, { apiKey: 'fixture-secret', maxRetries: 9 });
    assert.equal(result.stopReason, 'error'); assert.equal(calls, 1);
    if (status === 200) assert.match(result.errorMessage, /returned no image/);
  }
});

test('official OpenRouter mixed URL/data image outputs preserve provider image order', async () => {
  const adapter = makeImageProvider(provider, { body: {} }, async () => json({ choices: [{ message: { content: 'ordered images', images: [
    { image_url: { url: 'https://93.184.216.34/A.png' } },
    { image_url: { url: `data:image/png;base64,${Buffer.from('B').toString('base64')}` } },
    { image_url: { url: 'https://93.184.216.34/C.png' } },
  ] } }] }), async (url) => new Response(url.pathname.slice(1, 2), { headers: { 'content-type': 'image/png' } }));
  const result = await adapter.generateImages({ ...model, api: 'openrouter-images', baseUrl: provider.base_url }, context, { apiKey: 'fixture-secret' });
  assert.equal(result.stopReason, 'stop', result.errorMessage);
  assert.deepEqual(result.output.filter((part) => part.type === 'image').map((part) => Buffer.from(part.data, 'base64').toString()), ['A', 'B', 'C']);
  assert.equal(result.output[0].text, 'ordered images');
});

test('image download cancellation settles during pending DNS and late DNS cannot open a socket', async () => {
  const { publicImageFetch } = await import('../image-provider.mjs');
  const controller = new AbortController(); let releaseDns, sockets = 0;
  const dns = new Promise((resolve) => { releaseDns = resolve; });
  const pending = publicImageFetch('https://cdn.invalid/image', { signal: controller.signal, lookup: () => dns,
    request: () => { sockets++; throw new Error('late DNS must not open a socket'); } });
  const reason = new Error('image request canceled during DNS'); controller.abort(reason);
  let timeout;
  try {
    await assert.rejects(Promise.race([pending, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error('DNS abort did not settle promptly')), 100); })]), (error) => error === reason);
  } finally { clearTimeout(timeout); }
  assert.equal(sockets, 0);
  releaseDns([{ address: '93.184.216.34', family: 4 }]);
  await Promise.resolve(); await Promise.resolve();
  assert.equal(sockets, 0);
});
