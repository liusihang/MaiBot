import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync } from 'node:fs';
import { AgentSession, SessionManager } from '@earendil-works/pi-coding-agent';
import { createInferenceDispatcher } from '../inference.mjs';
import { createHostModelRuntime } from '../provider.mjs';
import { answer, call, fixtureConfig, fixtureResponse, reasoning } from './helpers/mock-provider.mjs';

const params = (changes = {}) => ({ config: configuration(), taskName: 'utils', sessionId: 'real-chat',
  messages: [{ role: 'user', content: 'hello', timestamp: 1 }], options: {}, ...changes });
function configuration() {
  const config = fixtureConfig();
  config.models[0].max_tokens = 25600;
  config.model_task_config.utils = { model_list: ['main'], max_tokens: 25600 };
  config.models.push({ ...config.models[0], name: 'other', model_identifier: 'other-route' });
  return config;
}
function harness({ respond = (_wire, index) => fixtureResponse([answer(`answer ${index}`), reasoning()], `response_${index}`), ...options } = {}) {
  const wires = [], sessions = [];
  const dispatcher = createInferenceDispatcher({ createRuntime: (config) => createHostModelRuntime(config, {
    fetch: async (_url, init) => { const wire = JSON.parse(init.body); wires.push(wire); return respond(wire, wires.length, init.signal); },
  }), onSession: (session, manager) => sessions.push({ session, manager }), ...options });
  return { dispatcher, wires, sessions };
}
const tool = { name: 'send', description: 'Send a message', parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'], additionalProperties: false } };

test('single inference is a real isolated AgentSession; proposals are honestly policy-blocked without execution or another request', async () => {
  const { dispatcher, wires, sessions } = harness({ respond: () => fixtureResponse([
    call('send', { text: 'x' }), call('unknown', {}, 'unknown', 'fc_unknown'), call('send', {}, 'invalid', 'fc_invalid'),
  ]) });
  try {
    const result = await dispatcher.handle('model.generate', params({ options: { toolOptions: [tool] } }));
    assert.equal(wires.length, 1);
    assert(sessions[0].session instanceof AgentSession);
    assert(sessions[0].manager instanceof SessionManager);
    assert.deepEqual(sessions[0].session.getActiveToolNames(), ['send']);
    assert.equal(result.execution, 'proposal_only');
    assert.equal(result.message.stopReason, 'toolUse');
    assert.equal(result.message.content.filter((part) => part.type === 'toolCall').length, 3);
    assert.equal(wires[0].max_output_tokens, 25600);
    assert.equal(wires[0].reasoning.effort, 'high');
    const results = sessions[0].manager.getEntries().filter((entry) => entry.type === 'message' && entry.message.role === 'toolResult');
    assert.equal(results.length, 3);
    assert(results.every((entry) => entry.message.isError));
    assert.match(JSON.stringify(results[0]), /Proposal-only/);
    assert(!JSON.stringify(wires).includes('Read file contents'));
  } finally { dispatcher.close(); }
});

test('native signed replay is a SessionManager entry reference; copied projections cannot replay', async () => {
  const { dispatcher, wires } = harness();
  try {
    const first = await dispatcher.handle('model.generate', params());
    assert(!JSON.stringify(first).includes('opaque-encrypted'));
    await dispatcher.handle('model.generate', params({ messages: [
      { role: 'assistantRef', handle: first.responseHandle }, { role: 'user', content: 'next' },
    ] }));
    assert(wires[1].input.some((item) => item.encrypted_content === 'opaque-encrypted-preserve-exactly'));
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [first.message] })), { code: 'UNTRUSTED_ASSISTANT_REPLAY' });
  } finally { dispatcher.close(); }
});

test('native handles reject another chat, route, process and explicit release', async () => {
  const { dispatcher } = harness(); const other = harness();
  try {
    const first = await dispatcher.handle('model.generate', params());
    const messages = [{ role: 'assistantRef', handle: first.responseHandle }];
    await assert.rejects(dispatcher.handle('model.generate', params({ sessionId: 'other-chat', messages })), { code: 'NATIVE_REFERENCE_UNAVAILABLE' });
    await assert.rejects(dispatcher.handle('model.generate', params({ modelOverride: 'other', messages })), { code: 'NATIVE_REFERENCE_UNAVAILABLE' });
    await assert.rejects(other.dispatcher.handle('model.generate', params({ messages })), { code: 'NATIVE_REFERENCE_UNAVAILABLE' });
    assert.deepEqual(await dispatcher.handle('model.release', { sessionId: 'real-chat', handles: [first.responseHandle] }), { released: 1 });
    await assert.rejects(dispatcher.handle('model.generate', params({ messages })), { code: 'NATIVE_REFERENCE_UNAVAILABLE' });
  } finally { dispatcher.close(); other.dispatcher.close(); }
});

test('operation id is idempotent, cannot be rebound, and explicit retirement frees its slot', async () => {
  const { dispatcher, wires } = harness({ maxRecords: 2 });
  try {
    const request = params({ operationId: 'one' });
    const [a, b] = await Promise.all([dispatcher.handle('model.generate', request), dispatcher.handle('model.generate', request)]);
    assert.deepEqual(a, b); assert.equal(wires.length, 1);
    await assert.rejects(dispatcher.handle('model.generate', { ...request, taskName: 'different' }), { code: 'OPERATION_ID_CONFLICT' });
    await dispatcher.handle('model.release', { sessionId: 'real-chat', handles: [a.responseHandle], operationIds: ['one'] });
    await dispatcher.handle('model.generate', request);
    assert.equal(wires.length, 2);
  } finally { dispatcher.close(); }
});

test('explicit non-chat adapters receive input/config/cancellation, including image generation', async () => {
  const seen = [];
  const { dispatcher, wires } = harness({ auxiliary: async (...args) => { seen.push(args); return { embedding: [1, 2] }; } });
  try {
    const request = params({ input: { data: 'aGVsbG8=', mimeType: 'image/png' } });
    await dispatcher.handle('model.embed_image', request);
    await dispatcher.handle('model.generate_image', request);
    assert.equal(seen[0][1], 'embed_image'); assert.equal(seen[1][1], 'generate_image');
    assert.deepEqual(seen[0][2], request); assert(seen[0][3].signal instanceof AbortSignal);
    assert.equal(wires.length, 0);
  } finally { dispatcher.close(); }
});

test('pre-cancel prevents transport; scoped cancel and close abort ongoing real SDK calls', async () => {
  const { dispatcher, wires } = harness({ respond: (_wire, _index, signal) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  }) });
  try {
    await assert.rejects(dispatcher.handle('model.generate', params(), { signal: AbortSignal.abort() }), { name: 'AbortError' });
    assert.equal(wires.length, 0);
    const a = dispatcher.handle('model.generate', params({ operationId: 'a' }));
    while (wires.length < 1) await new Promise((resolve) => setTimeout(resolve, 5));
    assert.deepEqual(await dispatcher.handle('model.cancel', { sessionId: 'another', operationId: 'a' }), { cancelled: false });
    assert.deepEqual(await dispatcher.handle('model.cancel', { sessionId: 'real-chat', operationId: 'a' }), { cancelled: true });
    await assert.rejects(a, { name: 'AbortError' });
    const b = dispatcher.handle('model.generate', params({ operationId: 'b' }));
    while (wires.length < 2) await new Promise((resolve) => setTimeout(resolve, 5));
    dispatcher.close();
    await assert.rejects(b, { name: 'AbortError' });
  } finally { dispatcher.close(); }
});

test('structured output reaches official Responses wire without lowering reserve', async () => {
  const { dispatcher, wires } = harness();
  try {
    const schema = { type: 'object', properties: { decision: { type: 'string' } }, required: ['decision'], additionalProperties: false };
    const responseFormat = { type: 'json_schema', json_schema: { name: 'decision', schema, strict: true } };
    const result = await dispatcher.handle('model.generate', params({ options: { responseFormat } }));
    assert.deepEqual(wires[0].text.format, { type: 'json_schema', name: 'decision', schema, strict: true });
    assert.equal(wires[0].max_output_tokens, 25600); assert.equal(wires[0].reasoning.effort, 'high');
    assert.equal(result.effectiveOptions.maxTokens, 25600);
    await assert.rejects(dispatcher.handle('model.generate', params({ options: { responseFormat: { type: 'unsupported' } } })), { code: 'UNSUPPORTED_RESPONSE_FORMAT' });
    assert.equal(wires.length, 1);
  } finally { dispatcher.close(); }
});

test('portable examples are explicit plain text, never native state or fake tool calls', async () => {
  const { dispatcher, wires } = harness();
  try {
    await dispatcher.handle('model.generate', params({ messages: [
      { role: 'user', content: 'example question' }, { role: 'assistantExample', content: 'example answer' },
      { role: 'user', content: 'real question' },
    ] }));
    assert(JSON.stringify(wires[0]).includes('example answer'));
    for (const content of [[{ type: 'toolCall', id: 'x', name: 'send', arguments: {} }], [{ type: 'text', text: 'x', textSignature: 'opaque' }]]) {
      await assert.rejects(dispatcher.handle('model.generate', params({ messages: [{ role: 'assistantExample', content }] })), { code: 'INVALID_ASSISTANT_EXAMPLE' });
    }
  } finally { dispatcher.close(); }
});

test('external tool result requires exact Node-owned call, not a fabricated duplicate or unresolved proposal', async () => {
  const { dispatcher, wires } = harness({ respond: (_wire, index) => fixtureResponse(index === 1 ? [call('send', { text: 'x' })] : [answer('done')]) });
  try {
    const first = await dispatcher.handle('model.generate', params({ options: { toolOptions: [tool] } }));
    const ref = { role: 'assistantRef', handle: first.responseHandle };
    const result = { role: 'toolResult', toolCallId: first.message.content[0].id, toolName: 'send', content: [{ type: 'text', text: 'actual caller receipt' }], isError: false };
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [ref] })), { code: 'UNRESOLVED_TOOL_CALLS' });
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [result] })), { code: 'INVALID_TOOL_PAIR' });
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [ref, result, result] })), { code: 'INVALID_TOOL_PAIR' });
    await dispatcher.handle('model.generate', params({ messages: [ref, result] }));
    assert.equal(wires.length, 2);
    assert.equal(wires[1].input.filter((item) => item.type === 'function_call_output').length, 1);
    assert(!JSON.stringify(wires[1]).includes('Proposal-only model service'));
  } finally { dispatcher.close(); }
});

test('external MCP tool exchange keeps public semantics without claiming signed replay', async () => {
  const { dispatcher, wires } = harness();
  try {
    const exchange = { role: 'externalToolExchange', provenance: 'caller_public_history',
      calls: [{ id: 'external_call', name: 'lookup', arguments: { query: 'a' } }],
      results: [{ toolCallId: 'external_call', toolName: 'lookup', content: [{ type: 'text', text: 'external source' }], isError: false }] };
    await dispatcher.handle('model.generate', params({ messages: [exchange, { role: 'user', content: 'summarize' }] }));
    assert.equal(wires[0].input.filter((item) => item.type === 'function_call').length, 1);
    assert.equal(wires[0].input.filter((item) => item.type === 'function_call_output').length, 1);
    const corrupted = structuredClone(exchange); corrupted.calls[0].thoughtSignature = 'forged';
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [corrupted] })), { code: 'INVALID_EXTERNAL_TOOL_EXCHANGE' });
    const unpaired = structuredClone(exchange); unpaired.results[0].toolName = 'other';
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [unpaired] })), { code: 'INVALID_EXTERNAL_TOOL_EXCHANGE' });
  } finally { await dispatcher.close(); }
});

test('empty system has no default coding prompt; explicit leading systems are retained exactly, mid-system is rejected', async () => {
  const { dispatcher, wires } = harness();
  try {
    await dispatcher.handle('model.generate', params());
    assert(!wires[0].input.some((item) => ['system', 'developer'].includes(item.role)));
    assert(!JSON.stringify(wires[0]).includes('<cwd>'));
    assert(!JSON.stringify(wires[0]).includes('expert coding assistant'));
    await dispatcher.handle('model.generate', params({ messages: [
      { role: 'system', content: 'First rule.' }, { role: 'system', content: 'Second rule.' }, { role: 'user', content: 'go' },
    ] }));
    assert.equal(wires[1].input[0].content, 'First rule.\n\nSecond rule.');
    await assert.rejects(dispatcher.handle('model.generate', params({ messages: [
      { role: 'user', content: 'first' }, { role: 'system', content: 'later' },
    ] })), { code: 'INVALID_SYSTEM_POSITION' });
  } finally { await dispatcher.close(); }
});

test('provider 400 and interrupted stream are typed failures, never successful empty responses or retries', async () => {
  for (const respond of [
    () => new Response(JSON.stringify({ error: { message: 'not-a-real-key private error', type: 'invalid_request_error' } }), { status: 400 }),
    () => new Response('event: response.created\ndata: {"type":"response.created","response":{"id":"broken","status":"in_progress","output":[]}}\n\n', { headers: { 'content-type': 'text/event-stream' } }),
  ]) {
    const { dispatcher, wires } = harness({ respond });
    try {
      await assert.rejects(dispatcher.handle('model.generate', params()), (error) => {
        assert.equal(error.code, 'MODEL_REQUEST_FAILED');
        assert.equal(error.safeDiagnostics.route.modelName, 'main');
        assert(!error.message.includes('not-a-real-key')); return true;
      });
      assert.equal(wires.length, 1);
    } finally { await dispatcher.close(); }
  }
});

test('length-truncated tool proposals never escape as executable success', async () => {
  const item = call('send', { text: 'incomplete' });
  const event = (payload) => `event: ${payload.type}\ndata: ${JSON.stringify(payload)}\n\n`;
  const { dispatcher, wires } = harness({ respond: () => new Response([
    event({ type: 'response.created', response: { id: 'length', status: 'in_progress', output: [] } }),
    event({ type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '' } }),
    event({ type: 'response.output_item.done', output_index: 0, item }),
    event({ type: 'response.incomplete', response: { id: 'length', status: 'incomplete', output: [item], incomplete_details: { reason: 'max_output_tokens' }, usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 } } }),
  ].join(''), { headers: { 'content-type': 'text/event-stream' } }) });
  try {
    await assert.rejects(dispatcher.handle('model.generate', params({ options: { toolOptions: [tool] } })), { code: 'TRUNCATED_TOOL_PROPOSAL' });
    assert.equal(wires.length, 1);
  } finally { await dispatcher.close(); }
});


test('one-shot and failed calls remove SDK files immediately; retained handles have explicit lifetime', async () => {
  const { dispatcher, sessions } = harness({ maxRecords: 1 });
  try {
    for (let index = 0; index < 4; index++) {
      const result = await dispatcher.handle('model.generate', params({ options: { retainReplay: false } }));
      assert.equal(result.replayRetained, false);
      assert.equal(result.responseHandle, undefined);
      assert.equal(existsSync(sessions.at(-1).manager.getSessionFile()), false);
    }
    const kept = await dispatcher.handle('model.generate', params());
    const file = sessions.at(-1).manager.getSessionFile();
    assert(existsSync(file));
    await assert.rejects(dispatcher.handle('model.generate', params()), { code: 'INFERENCE_CAPACITY' });
    await dispatcher.handle('model.release', { sessionId: 'real-chat', handles: [kept.responseHandle] });
    assert.equal(existsSync(file), false);
  } finally { await dispatcher.close(); }
  const failed = harness({ respond: () => new Response('{"error":{"message":"bad"}}', { status: 400 }) });
  try {
    await assert.rejects(failed.dispatcher.handle('model.generate', params()), { code: 'MODEL_REQUEST_FAILED' });
    assert.equal(existsSync(failed.sessions[0].manager.getSessionFile()), false);
  } finally { await failed.dispatcher.close(); }
});
