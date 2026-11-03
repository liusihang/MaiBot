import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { validateSessionLog } from '../session-log.mjs';

function log(t, value) {
  const dir = mkdtempSync(join(tmpdir(), 'maibot-log-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const file = join(dir, 'session.jsonl');
  writeFileSync(file, value);
  return file;
}
const header = JSON.stringify({ type: 'session', id: 'session', version: 3 }) + '\n';
const entry = (id, parentId, text = '') => JSON.stringify({ type: 'message', id, parentId, message: { role: 'user', content: text } }) + '\n';

test('streaming validation crosses UTF-8 and record boundaries while preserving exact bytes', async (t) => {
  const text = '群聊图片正文🙂'.repeat(15000);
  const original = header + entry('a', null, text) + entry('b', 'a', text);
  const file = log(t, original);
  await validateSessionLog(file);
  assert.equal(readFileSync(file, 'utf8'), original);
});

test('streaming validation preserves the former trailing whitespace allowance', async (t) => {
  await validateSessionLog(log(t, header + entry('a', null) + ' \n\n'));
});

for (const [name, value] of [
  ['incomplete append', header + entry('a', null).trimEnd()],
  ['invalid JSON', header + '{broken}\n'],
  ['missing header', entry('a', null)],
  ['duplicate id', header + entry('a', null) + entry('a', 'a')],
  ['missing ancestor', header + entry('a', 'missing')],
  ['blank record between entries', header + '\n' + entry('a', null)],
]) {
  test(`streaming validation rejects ${name} without repairing the source`, async (t) => {
    const file = log(t, value);
    await assert.rejects(validateSessionLog(file), { code: 'CORRUPT_SESSION' });
    assert.equal(readFileSync(file, 'utf8'), value);
  });
}
