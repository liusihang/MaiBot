# MaiBot Pi sidecar

This is a transport adapter around the **installed official `Agent` class** from
`@earendil-works/pi-agent-core@1.4.0`, not a second agent loop. Both Pi packages are
exact-pinned in `package.json` and `package-lock.json`. Node **22.19.0 or later** is
required. Python keeps MaiBot's existing provider/model selection, persona, QQ
permissions, tool implementations, durable ledger and subagent ownership.

```sh
npm --prefix runtime/pi_agent ci --ignore-scripts --no-audit --no-fund
npm --prefix runtime/pi_agent test
node runtime/pi_agent/sidecar.mjs
```

The tests use real installed Pi packages and mock the Python callbacks; they do
not call a real model, QQ service, or other network service. Missing Node or Pi
dependencies is a startup error, never a reason to fall back to a bespoke loop.

## JSONL protocol

stdin and stdout carry one JSON object per line; stderr is diagnostic-only. The
input pump remains live while a prompt runs, so independent groups, callback
responses, arriving messages and cancellation can be processed concurrently.

Python command:

```json
{"id":"host-1","method":"init","params":{"sessionId":"real-chat-session-id","system":"Frozen MaiBot persona","tools":[],"messages":[],"revision":0}}
```

Node reply:

```json
{"id":"host-1","result":{"sessionId":"real-chat-session-id","revision":0,"messages":[],"tools":[],"queuedMessages":[],"isRunning":false}}
```

The example reply is abbreviated. Errors use
`{"id":"host-1","error":{"code":"SESSION_BUSY","message":"..."}}`.
Command IDs must be unique while in flight. The `cb-` prefix is reserved for
Node callbacks. Responses can arrive out of order and **must be correlated by
ID**, not by the last command issued.

Every command takes `sessionId`:

| Method | Additional parameters | Behavior |
| --- | --- | --- |
| `init` | `system`, `tools?`, `messages?`, `revision?` | Construct exactly one Pi Agent. Repeated init is rejected. `messages` is the full Pi replay view, including opaque fields. |
| `prompt` | `additions: user[]` | Append new messages and await the Pi run. If already running, persist and queue them instead, returning `queued: true`. |
| `append` | `additions: user[]` | Persist arrivals to the steering queue without starting a run. |
| `continue` | — | Resume queued input, or a context ending in user/toolResult. It does not rewrite a failed assistant response into a user message. |
| `state` | — | Return a copy of state without starting work. |
| `add_tools` | `tools: declaration[]` | Append previously unknown names. A same-name identical declaration is a no-op; redefining or removing tools is rejected. |
| `replace_context` | `expectedRevision`, `messages`, `reason` | **Explicit compaction only**, idle, settled and without queued arrivals. Compare-and-swap must match. Preserve the fixed system and full current tool declarations. |
| `resolve_tools` | `expectedRevision`, `results: toolResult[]` | Explicit host recovery only. Append results matching interrupted, unresolved call IDs after the host has established their actual outcomes. Never executes a tool. |
| `abort` | — | Signal cancellation, await Pi idle and terminal event acknowledgments. |
| `close` | — | Abort/settle, persist `session_closed`, remove this Agent. Does not delete Python history. |

The state also contains `system`, `pendingToolCalls`, `error`, `poisoned`,
`paused`, `unresolvedToolCalls`, `maintenanceRequired`, `continuationRequired` and
`terminatedByTool`. `isRunning` stays true until awaited Pi event
listeners settle. A returned snapshot does **not** necessarily mean a successful
model response: inspect `error` and the final assistant `stopReason`.

### Callbacks

Node sends a request such as:

```json
{"id":"cb-1","method":"model","params":{"sessionId":"group-a","system":"Frozen persona","messages":[],"tools":[]}}
```

Python returns `{"id":"cb-1","result":...}` or
`{"id":"cb-1","error":{"code":"HOST_ERROR","message":"..."}}`.

* **`model`**: receives the complete Pi transcript and current tool schemas.
  Return `{ "message": { "role": "assistant", "content": [...],
  "stopReason": "toolUse", "native": { ... } } }`. The wrapper also accepts
  the assistant object directly. `content` is a minimal Pi projection of text,
  thinking and tool-call blocks, used only to drive Pi's loop. Tool calls use
  `{type:"toolCall", id, name, arguments}`. Optional missing Pi bookkeeping
  fields get synthetic defaults; the Python-selected provider remains the real
  provider. No provider adapter, API key or model discovery is invoked in Node.
* **`tool`**: receives `{sessionId, toolCallId, name, arguments}`. Return
  `{content:[{type:"text",text:"..."}],details:{...},isError?:false,
  terminate?:false}`. Throw/return an RPC error for execution failures.
  `terminate:true` ends this run after the current completed tool; remaining
  calls in its batch are recorded as blocked, not dispatched. Use this for
  intentional wait/stop tools. On-demand subagents are ordinary host tools.
* **`event`**: receives `{sessionId, revision, event}`. The reply is a **durable
  commit acknowledgment**, not merely receipt. All Pi events are awaited before
  Pi proceeds. In particular, assistant `message_end` is committed before tool
  preflight; a toolResult `message_end` is committed before another sequential
  tool or model request. `agent_end` is acknowledged before the run settles.

`native` is opaque JSON, for example Python's tagged native codec
`{items:[...]}`. Node preserves it verbatim through live state, event callbacks,
replay hydration, and every subsequent model request. Pi's content projection
must **never** be used to reconstruct earlier native responses. For tool output,
put canonical native content in **`details.native`**, not a top-level field:
Pi itself builds toolResult messages from selected result fields and preserves
`details`. Native assistant output and native tool output are therefore distinct
persisted objects; the Python provider projection reads both.

### Pi 1.4 transcript details

The npm package's actual API is authoritative. Pi 1.4 puts the base prompt and
initial tool declarations in a leading `role:"system"` message with timestamp
zero. That generated baseline is returned by `init`; the host must persist it
before its first run. Later tool additions become **new** system messages with
empty `content` and `toolsAdded`, never edits to the earlier system message.
Python must not duplicate the base prompt or serialize declaration-only system
messages as ordinary user chat. No automatic pruning, history summarization,
section replacement or prompt rewriting occurs here.

Each ordinary run extends the existing message list. `replace_context` is the
only active-view replacement operation and is reserved before its first await,
so an acknowledged host compaction cannot race a local prompt and leave the two
sides with different views. Commands racing that reservation fail explicitly;
Python retains their durable arrivals and retries once it is safe. The ledger
keeps old events and the compaction checkpoint even when the active view shrinks.

### Arrivals and persistence reentrancy

External additions must be `role:"user"`. Their metadata, including
`hostEventId` and `native`, survives unchanged. The host should durably store an
incoming group message **before** submitting it. `append` and busy `prompt` emit
`event.type = "input_queued"` with `messages`, then add them to Pi's queue after
acknowledgment. Pi later emits normal user `message_end` events when it consumes
them. These are two lifecycle stages of the same input, **not two user messages**;
the host deduplicates using its stable event IDs.

Event revisions are reserved synchronously and increase monotonically. Callback
acknowledgments are not globally serialized: after committing an event, Python
may ingest another durable group arrival by issuing and awaiting `append` from
inside that event callback. This can produce a nested `input_queued` callback.
The Python reader must keep dispatching callbacks while a handler awaits a
command. The tests cover this exact round trip. Do not synchronously await
`abort`/`close` from inside the event being acknowledged: settlement naturally
requires that same acknowledgment; acknowledge first and issue control outside
that handler.

Extra events are `input_queued`, `tools_added`, `context_replaced`,
`run_aborted` and `session_closed`. Recovery message events carry
`recovery:true`. `revision` counts emitted event transactions, not just chat
messages. Hydration can pass the previous runtime revision through `init`.

### Cancellation, unknown outcomes and failures

A pending model/tool callback receives a cancellation notification with **no ID**:

```json
{"method":"cancel","params":{"callbackId":"cb-17","sessionId":"group-a"}}
```

Python cancels or reconciles the correlated task. Node ignores late callback
responses. Event acknowledgments are deliberately not canceled: terminal state
still needs durable recording. A pre-aborted model request never calls Python.

Pi's sequential cancellation can stop between calls in a multi-call assistant
message. The unanswered IDs remain in `unresolvedToolCalls` and the idle session
becomes `paused`; another `prompt`/`continue` is rejected. New arrivals may still
be durably queued. Only explicit host recovery may close unanswered IDs, using
`resolve_tools` with known `not_started`, `confirmed`, or another truthful
outcome. An RPC cancellation is **not evidence that an external side effect did
not happen**; Python must preserve `unknown` outcomes and must not replay QQ
sends or other irreversible actions automatically. Never fabricate a success to
clear the pause. Host-side tool outcome safety remains authoritative, including
for a canceled call for which Pi emitted an error result.

If persistence fails, the session is poisoned and aborts before another tool or
model dispatch. Subsequent writes fail closed. Repair the durable host state and
explicitly restart; do not continue the potentially uncommitted in-memory view.

### Explicit context maintenance

The Python provider may set `message.hostMaintenanceRequired = true` based on
reported provider usage and the host's configured soft-pressure policy. Pi
finishes **every tool in that assistant batch**, persists their results, then
ends at `finishTurn`; it does not abort a partially executed batch. State returns
`maintenanceRequired:true`, and another prompt/continuation is rejected until a
successful explicit `replace_context`. `init` accepts `maintenanceRequired` for
recovery of this control state. The host may retain newly arrived input durably
outside the Pi queue until maintenance finishes; the replacement command still
requires no queued input. A soft threshold is not proof of provider capacity;
Python's exact request-admission check remains mandatory and may fail closed.

After maintenance, the host resumes an interrupted automatic tool follow-up only
when the returned pre-compaction `continuationRequired` was true. A final plain
answer or a terminating wait/stop tool sets it false, preventing an unsolicited
duplicate answer. `terminatedByTool` also reports the latter case. Both
maintenance flags clear on successful context replacement. A new `prompt` after
a terminating tool preserves FIFO: earlier retained queue entries are consumed
before the newly supplied additions, not after them.

### Host control is not provider output

If **verified** request admission rejects a request as `over_limit` before any
provider call, Python may return this control result instead of a model message:

```json
{"maintenanceRequired":true,"hostControl":"capacity_admission","reason":"Verified request exceeds known capacity"}
```

The bridge records an explicitly labeled host-control receipt with
`hostControl:"capacity_admission"`, `hostControlReason`, empty `content`, empty
`native.items`, `hostMaintenanceRequired:true` and
`hostContinuationRequired:true`. It is not represented as actual provider
reasoning/text. Pi's stream protocol requires an assistant-shaped terminal
receipt, but the empty native items contribute **nothing** to provider replay.
The existing transcript is preserved, and Pi pauses for explicit host compaction
before retrying. The host may alternatively return that same marked receipt in
`result.message`. Unknown capacity must remain a host failure, not an invented
`over_limit` decision. The host must bound maintenance retries and detect a
non-shrinking request; this bridge never truncates an input or loops on capacity.

After explicit compaction, a retained host-control receipt may still be the last
Pi message. For this exact, marked boundary only, `continue` uses official
`Agent.prompt([])` to resume without adding an invented user message; ordinary
assistant tails still follow official `Agent.continue()` restrictions.

A callback transport/provider exception has a different audit marker:
`hostSynthetic:"provider_callback_failure"`, `hostControl:"provider_failure"`,
empty `native.items`, and the
truthful `error`/`aborted` stop reason. Only the bridge's own catch-generated
receipt receives this marker. It preserves the failure in the ledger without
pretending the provider emitted it, and does not poison provider replay for the
next legitimate input. A missing native field on an actual provider output is
**not** repaired by reconstructing it from Pi content. Uncertain external tool
outcomes remain subject to the separate host execution fence.
