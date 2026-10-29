/** Pi image operations for existing MaiBot endpoints, not chat-model workarounds. */
import { lookup } from 'node:dns/promises';
import { request as httpsRequest } from 'node:https';
import { isIP } from 'node:net';
import { Readable } from 'node:stream';
import { generateImages as openrouterImages } from '@earendil-works/pi-ai/api/openrouter-images';

const fail = (code, message) => Object.assign(new Error(message), { code });
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
function publicAddress(address) {
  if (isIP(address) === 4) {
    const [a, b, c] = address.split('.').map(Number);
    return !(a === 0 || a === 10 || a === 127 || a >= 224 || (a === 100 && b >= 64 && b <= 127)
      || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && (b === 168 || b === 0 || (b === 88 && c === 99)))
      || (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100))) || (a === 203 && b === 0 && c === 113));
  }
  if (isIP(address) !== 6 || !/^[23]/.test(address)) return false;
  const [first, second] = address.split(':').map((part) => parseInt(part || '0', 16));
  // Conservative native global-unicast subset: reject translation, 6to4,
  // protocol-assignment, documentation and former 6bone ranges.
  return first !== 0x2002 && first !== 0x3ffe && first !== 0x3fff
    && !(first === 0x2001 && (second < 0x200 || second === 0xdb8));
}
function validateImageUrl(raw) {
  const url = new URL(raw);
  const hostname = url.hostname.replace(/^\[|\]$/g, '');
  if (url.protocol !== 'https:' || url.username || url.password || (url.port && url.port !== '443')
      || hostname.toLowerCase() === 'localhost' || hostname.endsWith('.localhost')
      || (isIP(hostname) && !publicAddress(hostname))) throw fail('UNSAFE_IMAGE_URL', 'Image downloads require a public HTTPS URL');
  return url;
}
/** DNS itself is not cancelable, but the owning request must stop waiting and
 * never open a socket if the result arrives after cancellation. */
async function waitForDns(pending, signal) {
  if (!signal) return pending;
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
  try { return await Promise.race([pending, aborted]); }
  finally { signal.removeEventListener('abort', onAbort); }
}

/** DNS is resolved and checked once, then pinned into HTTPS lookup, avoiding rebinding. */
export async function publicImageFetch(rawUrl, { signal, lookup: lookupImpl = lookup, request: requestImpl = httpsRequest } = {}) {
  const url = validateImageUrl(rawUrl);
  const hostname = url.hostname.replace(/^\[|\]$/g, '');
  signal?.throwIfAborted();
  const addresses = isIP(hostname) ? [{ address: hostname, family: isIP(hostname) }] : await waitForDns(lookupImpl(hostname, { all: true }), signal);
  if (!addresses.length || addresses.some(({ address }) => !publicAddress(address))) throw fail('UNSAFE_IMAGE_URL', 'Image URL resolves to a non-public address');
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const req = requestImpl(url, { signal, headers: { accept: 'image/*' }, lookup: (_hostname, options, callback) => {
      if (options.all) callback(null, addresses); else callback(null, addresses[0].address, addresses[0].family);
    } }, (res) => {
      try {
        const headers = new Headers();
        for (const [key, value] of Object.entries(res.headers)) if (value != null) headers.set(key, Array.isArray(value) ? value.join(', ') : value);
        const noBody = [204, 205, 304].includes(res.statusCode);
        if (noBody) res.destroy();
        // Node's stream bridge keeps backpressure and destroys the socket when
        // a caller cancels an error/oversize/redirect body.
        resolve(new Response(noBody ? null : Readable.toWeb(res), { status: res.statusCode, headers }));
      } catch (error) { res.destroy(); reject(error); }
    });
    req.on('error', reject); req.end();
  });
}
async function downloadImage(rawUrl, signal, injectedFetch) {
  let url = validateImageUrl(rawUrl);
  for (let redirect = 0; redirect <= 3; redirect++) {
    signal?.throwIfAborted();
    // Never use createProviderFetch: no provider auth, query, cookies or headers
    // may follow a returned URL, including same-origin URLs and redirects.
    const response = await (injectedFetch ?? publicImageFetch)(url, { signal, redirect: 'manual', credentials: 'omit', headers: { accept: 'image/*' } });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      await response.body?.cancel();
      if (!response.headers.get('location') || redirect === 3) throw fail('IMAGE_DOWNLOAD_ERROR', 'Image download redirect limit exceeded');
      url = validateImageUrl(new URL(response.headers.get('location'), url)); continue;
    }
    if (!response.ok) { await response.body?.cancel(); throw fail('IMAGE_DOWNLOAD_ERROR', `Image download HTTP ${response.status}`); }
    const mimeType = response.headers.get('content-type')?.split(';')[0].trim().toLowerCase();
    if (!/^image\/(png|jpeg|webp|gif|avif)$/.test(mimeType ?? '')) { await response.body?.cancel(); throw fail('INVALID_IMAGE_RESPONSE', 'Image download has an unsupported MIME type'); }
    if (Number(response.headers.get('content-length')) > MAX_IMAGE_BYTES) { await response.body?.cancel(); throw fail('IMAGE_TOO_LARGE', 'Image download exceeds the size limit'); }
    const chunks = []; let size = 0;
    for await (const chunk of response.body ?? []) {
      signal?.throwIfAborted(); size += chunk.length;
      if (size > MAX_IMAGE_BYTES) throw fail('IMAGE_TOO_LARGE', 'Image download exceeds the size limit');
      chunks.push(Buffer.from(chunk));
    }
    if (!size) throw fail('INVALID_IMAGE_RESPONSE', 'Image download is empty');
    return { type: 'image', data: Buffer.concat(chunks).toString('base64'), mimeType };
  }
}
function usageFor(raw, model) {
  const value = raw.usage ?? raw.usageMetadata;
  if (!value) return undefined;
  const cacheRead = value.input_tokens_details?.cached_tokens ?? value.prompt_tokens_details?.cached_tokens ?? value.cachedContentTokenCount ?? 0;
  const input = Math.max(0, (value.input_tokens ?? value.prompt_tokens ?? value.promptTokenCount ?? 0) - cacheRead);
  const output = value.output_tokens ?? value.completion_tokens ?? ((value.candidatesTokenCount ?? 0) + (value.thoughtsTokenCount ?? 0));
  const usage = { input, output, cacheRead, cacheWrite: 0, totalTokens: value.total_tokens ?? value.totalTokenCount ?? input + output + cacheRead,
    cost: { input: input * (model.cost?.input ?? 0) / 1e6, output: output * (model.cost?.output ?? 0) / 1e6,
      cacheRead: cacheRead * (model.cost?.cacheRead ?? 0) / 1e6, cacheWrite: 0, total: 0 } };
  usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead;
  return usage;
}

export function makeImageProvider(provider, extra, fetcher, downloadFetch, capability = {}) {
  return { generateImages: async (model, context, options = {}) => {
    const result = { api: model.api, provider: model.provider, model: model.id, output: [], stopReason: 'stop', timestamp: Date.now() };
    const signal = AbortSignal.any([...(options.signal ? [options.signal] : []), AbortSignal.timeout((provider.timeout ?? 240) * 1000)]);
    try {
      signal.throwIfAborted();
      if (capability.mode === 'strict') throw fail('CONTEXT_CAPABILITY_UNKNOWN', 'Strict capacity admission cannot establish a safe image-generation operation bound');
      const transport = options.fetch ?? fetcher;
      const params = { ...extra.body, ...options.metadata };
      for (const key of ['model', 'prompt', 'contents', 'messages', 'input', 'image', 'image[]', 'stream']) {
        if (key in params) throw fail('RESERVED_MODEL_PARAMETER', `Image parameters cannot override identity/input field: ${key}`);
      }
      if (model.api === 'openrouter-images') {
        const imageOrder = [];
        // The official provider preserves data URLs only. Observe (do not
        // rewrite) its HTTP response to recover URL-only image outputs through
        // the same isolated, unauthenticated downloader as other image APIs.
        const routerFetch = async (input, init) => {
          const response = await transport(input, init);
          if (response.ok) {
            const raw = await response.clone().json();
            for (const image of raw.choices?.[0]?.message?.images ?? []) {
              const url = typeof image.image_url === 'string' ? image.image_url : image.image_url?.url;
              if (typeof url === 'string') {
                if (!url.startsWith('data:')) imageOrder.push({ url });
                else if (/^data:([^;]+);base64,(.+)$/.test(url)) imageOrder.push({ nativeData: true });
              }
            }
          }
          return response;
        };
        Object.assign(result, await openrouterImages(model, context, { ...options, signal, fetch: routerFetch, maxRetries: 0,
          onPayload: async (payload, activeModel) => {
            let merged = { ...payload, ...params };
            if (options.onPayload) merged = (await options.onPayload(merged, activeModel)) ?? merged;
            if (merged.model !== model.id) throw fail('RESERVED_MODEL_PARAMETER', 'Image hook cannot replace the selected model');
            return merged;
          } }));
        if (result.stopReason !== 'stop') return result;
        const nativeImages = result.output.filter((item) => item.type === 'image');
        const orderedImages = [];
        for (const entry of imageOrder) {
          if (entry.nativeData) {
            const image = nativeImages.shift();
            if (!image) throw fail('INVALID_IMAGE_RESPONSE', 'Pi image output order could not be established');
            orderedImages.push(image);
          } else orderedImages.push(await downloadImage(entry.url, signal, downloadFetch));
        }
        if (nativeImages.length) throw fail('INVALID_IMAGE_RESPONSE', 'Pi image output order could not be established');
        result.output = [...result.output.filter((item) => item.type !== 'image'), ...orderedImages];
        if (!result.output.some((item) => item.type === 'image')) throw fail('INVALID_IMAGE_RESPONSE', 'Pi image provider returned no image');
        signal.throwIfAborted();
        return result;
      }
      const prompt = context.input.filter((item) => item.type === 'text').map((item) => item.text).join('\n');
      const images = context.input.filter((item) => item.type === 'image');
      const headers = { authorization: `Bearer ${options.apiKey || provider.api_key}`, 'content-type': 'application/json' };
      let url, body;
      if (model.api === 'maibot-openai-images') {
        url = `${provider.base_url.replace(/\/$/, '')}/images/${images.length ? 'edits' : 'generations'}`;
        if (images.length) {
          body = new FormData(); body.set('model', model.id); body.set('prompt', prompt);
          for (const [i, image] of images.entries()) body.append(images.length > 1 ? 'image[]' : 'image', new Blob([Buffer.from(image.data, 'base64')], { type: image.mimeType }), `image-${i}.${image.mimeType.split('/')[1]}`);
          for (const [key, value] of Object.entries(params)) body.set(key, typeof value === 'string' ? value : JSON.stringify(value));
          delete headers['content-type'];
        } else body = { ...params, model: model.id, prompt };
      } else if (model.api === 'maibot-google-images') {
        url = `${provider.base_url.replace(/\/$/, '')}/models/${encodeURIComponent(model.id.replace(/^models\//, ''))}:generateContent`;
        headers['x-goog-api-key'] = options.apiKey || provider.api_key; delete headers.authorization;
        body = { contents: [{ role: 'user', parts: [{ text: prompt }, ...images.map((image) => ({ inlineData: { mimeType: image.mimeType, data: image.data } }))] }],
          generationConfig: { responseModalities: ['TEXT', 'IMAGE'], ...params } };
      } else throw fail('UNSUPPORTED_IMAGE_PROVIDER', `Unsupported Pi image API: ${model.api}`);
      if (options.onPayload) body = (await options.onPayload(body, model)) ?? body;
      if (model.api === 'maibot-openai-images' && (body instanceof FormData ? body.get('model') : body.model) !== model.id) throw fail('RESERVED_MODEL_PARAMETER', 'Image hook cannot replace the selected model');
      const response = await transport(url, { method: 'POST', headers, body: body instanceof FormData ? body : JSON.stringify(body), signal });
      await options.onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers.entries()) }, model);
      if (!response.ok) { await response.body?.cancel(); throw fail('MODEL_HTTP_ERROR', `Pi image provider HTTP ${response.status}`); }
      const raw = await response.json();
      if (model.api === 'maibot-openai-images') {
        for (const image of raw.data ?? []) {
          if (image.b64_json) result.output.push({ type: 'image', data: image.b64_json, mimeType: params.output_format === 'jpeg' ? 'image/jpeg' : params.output_format === 'webp' ? 'image/webp' : 'image/png' });
          else if (image.url) result.output.push(await downloadImage(image.url, signal, downloadFetch));
          if (image.revised_prompt) result.output.push({ type: 'text', text: image.revised_prompt });
        }
      } else for (const part of raw.candidates?.[0]?.content?.parts ?? []) {
        if (part.inlineData?.data) result.output.push({ type: 'image', data: part.inlineData.data, mimeType: part.inlineData.mimeType });
        else if (part.text) result.output.push({ type: 'text', text: part.text });
      }
      if (!result.output.some((item) => item.type === 'image')) throw fail('INVALID_IMAGE_RESPONSE', 'Pi image provider returned no image');
      signal.throwIfAborted();
      if (raw.id) result.responseId = raw.id;
      const usage = usageFor(raw, model); if (usage) result.usage = usage;
    } catch (error) {
      result.stopReason = signal.aborted ? 'aborted' : 'error';
      result.errorMessage = error.message;
    }
    return result;
  } };
}
