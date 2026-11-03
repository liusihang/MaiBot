import assert from 'node:assert/strict';
import test from 'node:test';
import { createImageProjection, IMAGE_PREVIEW_BYTES } from '../media-context.mjs';

test('large image projection preserves native messages and reuses deterministic previews', async () => {
  const image = { type: 'image', data: 'YQ=='.repeat(IMAGE_PREVIEW_BYTES / 4 + 1), mimeType: 'image/png' };
  const original = [{ role: 'user', content: [{ type: 'text', text: 'original' }, image] },
    { role: 'toolResult', toolCallId: 'call', content: [image] },
    { role: 'assistant', content: [{ type: 'thinking', thinking: 'native', signature: 'opaque' }] }];
  const snapshot = structuredClone(original); let calls = 0;
  const project = createImageProjection({ resize: async (_bytes, mime, options) => {
    calls++; assert.equal(mime, 'image/png'); assert.equal(options.maxBytes, IMAGE_PREVIEW_BYTES);
    return { data: 'cHJldmlldw==', mimeType: 'image/jpeg' };
  } });
  const first = await project(original), second = await project(original);
  assert.deepEqual(original, snapshot); assert.deepEqual(first, second); assert.equal(calls, 1);
  assert.equal(first[0].content[0], original[0].content[0]);
  assert.equal(first[0].content[1].data, 'cHJldmlldw==');
  assert.equal(first[1].toolCallId, 'call'); assert.equal(first[2], original[2]);
});

test('small images and plain text keep their existing payloads', async () => {
  const messages = [{ role: 'user', content: [{ type: 'image', data: 'small', mimeType: 'image/png' }] },
    { role: 'user', content: 'plain' }];
  const project = createImageProjection({ resize: async () => assert.fail('Small image should remain unchanged') });
  const result = await project(messages);
  assert.equal(result[0], messages[0]); assert.equal(result[1], messages[1]);
});

test('failed preview creation is explicit and preserves the original', async () => {
  const messages = [{ role: 'user', content: [{ type: 'image', data: 'a'.repeat(IMAGE_PREVIEW_BYTES + 1), mimeType: 'image/png' }] }];
  const project = createImageProjection({ resize: async () => null });
  await assert.rejects(project(messages), { code: 'IMAGE_PREVIEW_FAILED' });
  assert.equal(messages[0].content[0].data.length, IMAGE_PREVIEW_BYTES + 1);
});
