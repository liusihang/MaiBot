/** Strict append-log validation, bounded by one JSONL entry rather than file size. */
import { closeSync, createReadStream, fstatSync, openSync, readSync } from 'node:fs';
import { createInterface } from 'node:readline';

const corrupt = (message) => Object.assign(new Error(message), { code: 'CORRUPT_SESSION' });

export async function validateSessionLog(file) {
  const fd = openSync(file, 'r');
  try {
    const size = fstatSync(fd).size;
    const last = Buffer.alloc(1);
    if (!size || readSync(fd, last, 0, 1, size - 1) !== 1 || last[0] !== 10) {
      throw corrupt('Incomplete session append requires explicit recovery');
    }
  } finally { closeSync(fd); }

  const input = createReadStream(file, { encoding: 'utf8' });
  const lines = createInterface({ input, crlfDelay: Infinity });
  const ids = new Set();
  let header = true;
  let trailingWhitespace = false;
  try {
    for await (const line of lines) {
      // Match the former trimEnd(): trailing blank lines are harmless, while
      // an empty record between entries remains an invalid append log.
      if (!line.trim()) { trailingWhitespace = true; continue; }
      if (trailingWhitespace) throw corrupt('Invalid session JSONL requires explicit recovery');
      let entry;
      try { entry = JSON.parse(line); }
      catch { throw corrupt('Invalid session JSONL requires explicit recovery'); }
      if (header) {
        if (entry?.type !== 'session') throw corrupt('Session header is missing');
        header = false;
      } else {
        if (typeof entry?.id !== 'string' || ids.has(entry.id)
            || (entry.parentId !== null && !ids.has(entry.parentId))) {
          throw corrupt('Session entry IDs or ancestry are invalid');
        }
        ids.add(entry.id);
      }
    }
  } finally { lines.close(); input.destroy(); }
}
