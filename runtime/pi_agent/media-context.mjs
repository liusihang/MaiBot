/** Model-facing previews; native user/tool messages retain their original images. */
import { createHash } from 'node:crypto';
import { resizeImage } from '@earendil-works/pi-coding-agent';

export const IMAGE_PREVIEW_BYTES = 512 * 1024;

export function createImageProjection({ resize = resizeImage } = {}) {
  const cache = new Map();
  async function project(part, signal) {
    if (part.type !== 'image' || typeof part.data !== 'string' || part.data.length <= IMAGE_PREVIEW_BYTES) return part;
    signal?.throwIfAborted();
    const key = createHash('sha256').update(part.mimeType).update(part.data).digest('hex');
    if (!cache.has(key)) {
      if (cache.size >= 64) cache.delete(cache.keys().next().value);
      const pending = resize(Buffer.from(part.data, 'base64'), part.mimeType,
        { maxWidth: 2000, maxHeight: 2000, maxBytes: IMAGE_PREVIEW_BYTES, jpegQuality: 80 })
        .then((image) => {
          if (!image || image.data.length > IMAGE_PREVIEW_BYTES) {
            throw Object.assign(new Error('Image preview could not be prepared; original image retained.'),
              { code: 'IMAGE_PREVIEW_FAILED' });
          }
          return { data: image.data, mimeType: image.mimeType };
        });
      cache.set(key, pending);
      pending.catch(() => { if (cache.get(key) === pending) cache.delete(key); });
    }
    const preview = await cache.get(key);
    signal?.throwIfAborted();
    return { ...part, ...preview };
  }
  return async (messages, signal) => {
    const projected = [];
    for (const message of messages) {
      if (!['user', 'toolResult', 'custom'].includes(message.role) || !Array.isArray(message.content)) {
        projected.push(message); continue;
      }
      const content = [];
      for (const part of message.content) content.push(await project(part, signal));
      projected.push(content.every((part, i) => part === message.content[i]) ? message : { ...message, content });
    }
    return projected;
  };
}
