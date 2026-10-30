/** Real official Responses serializer/parser with an in-memory HTTP fixture, never a model endpoint. */
export function fixtureConfig() {
  return {
    api_providers: [{ name: 'offline', client_type: 'openai_responses', base_url: 'https://offline.invalid/v1', api_key: 'fixture-only' }],
    models: [{ name: 'main', api_provider: 'offline', model_identifier: 'exact-fixture-route', max_tokens: 1024,
      extra_params: { reasoning_effort: 'high' } }],
    model_task_config: { planner: { model_list: ['main'], max_tokens: 1024 } },
  };
}
export function fixtureResponse(items, id = 'resp_fixture') {
  const event = (value) => `event: ${value.type}\ndata: ${JSON.stringify(value)}\n\n`;
  const events = [{ type: 'response.created', response: { id, status: 'in_progress', output: [] } }];
  for (const [output_index, item] of items.entries()) {
    events.push({ type: 'response.output_item.added', output_index, item: { ...item, ...(item.type === 'function_call' ? { arguments: '' } : {}) } });
    events.push({ type: 'response.output_item.done', output_index, item });
  }
  events.push({ type: 'response.completed', response: { id, status: 'completed', output: items,
    usage: { input_tokens: 80, output_tokens: 20, total_tokens: 100 } } });
  return new Response(events.map(event).join(''), { headers: { 'content-type': 'text/event-stream' } });
}
export const answer = (text, id = 'msg_answer') => ({ type: 'message', id, role: 'assistant', status: 'completed', content: [{ type: 'output_text', text, annotations: [] }] });
export const reasoning = () => ({ type: 'reasoning', id: 'rs_opaque', encrypted_content: 'opaque-encrypted-preserve-exactly', summary: [{ type: 'summary_text', text: 'fixture summary' }] });
export const call = (name, args, id = 'call_qq', itemId = 'fc_qq') => ({ type: 'function_call', id: itemId, call_id: id, name, arguments: JSON.stringify(args), status: 'completed' });
