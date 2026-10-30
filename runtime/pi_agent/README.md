# MaiBot official Pi harness

Node owns provider execution and model conversation state through exact-pinned
`@earendil-works/pi-coding-agent@1.4.0` `createAgentSession`, `AgentSession` and
`SessionManager`. Python retains QQ/business tools, durable inbox/outbox receipts,
permissions and business stores. There is **no Python model callback**, copied
canonical Python transcript, planner/replyer loop, or fallback agent framework.

## Requirements

- Linux, Node **22.19.0+**, and **util-linux `flock`**, plus POSIX `sh` and `cat`.
- A local durable filesystem supporting advisory locks and `fsync`. Keep each
  group session directory on the same durable volume. Do not use an object-store
  mount with incompatible locking or sync semantics.
- `npm --prefix runtime/pi_agent ci --ignore-scripts --no-audit --no-fund`

`npm --prefix runtime/pi_agent test` uses real SDK code with mock transports or
localhost fake HTTP only. It does not spend model credits or send QQ messages.
Missing dependencies fail startup/session initialization; never fall back to
unlocked storage, Python inference, or a replacement custom loop.

## JSONL endpoints

stdin/stdout are duplex JSONL. Command IDs must be unique while in flight; `cb-`
is reserved for Node callbacks. The input pump stays live during execution.

| Method | Parameters beyond `sessionId` | Contract |
| --- | --- | --- |
| `init` | `sessionDir`, `system`, `tools`, `toolBindings`, private `config`, `taskName`, optional `summaryTaskName` | Acquire single-writer lock, strictly reopen SDK JSONL, verify frozen prompt/tools/bindings/route. Main task is `planner`; summary task defaults explicitly to `planner` (the prior sub-agent request mapping). |
| `append` | `additions` | Persist new user messages or pending SDK inbox entries, sync file and directory, then return `acceptedSourceIds`. |
| `prompt` | `references`, `additions` | Append tail changes and activate the official session. While running/compacting, persist pending input instead of losing it. |
| `continue` | optional `references`, `additions` | Explicit activation without inventing repeated user text. |
| `state` | — | Bounded status/conclusion/usage/queue counts/unresolved calls/scoped recent archive handles; never the native transcript. |
| `compact` | optional `expectedRevision`, `summary`, `summaryTaskName`, `firstKeptEntryId`, `preserveUnprocessed` | Explicit SDK `session.compact()` with a business summary hook. Maintenance uses `preserveUnprocessed:true`; it may return `compacted:false` when no completed prefix exists. |
| `resolve_tools` | known durable `results` | Reconcile orphan/unknown calls; never execute effects. Existing error outputs receive an appended receipt correction, not a replacement or duplicate function output. |
| `read_archive` | `eventIds`, `offset`, `limit` | Read public chat/tool/reference text only from this session. `eventIds` may include `pi-archive:…` handles printed in compaction summaries. |
| `abort`, `close` | — | Abort, settle durable terminal events, then release writer ownership. |

Each input needs a stable `hostEventId` (or `sourceMessageId`). ACKs identify
`hostEventId`. Deduplication searches the whole SDK archive, including compacted
messages; reusing a source with changed content is an error. References have
`namespace`, `sourceId`, `version`, `content`. Identical visible versions are not
injected twice; newer versions append, and compacted-away details can be retrieved
again. New time information belongs only in new tail messages.

Provider/service endpoints share this sidecar: `model.generate`,
`model.embed_text`, `model.embed_image`, `model.transcribe`, `model.generate_image`,
`model.release`, and `model.cancel`. Native response handles are Node-owned and
scoped. Chat generation uses an isolated official AgentSession; explicit non-chat
capability adapters are not fake chat requests. Closing the transport cancels
outstanding operations.

## Business callbacks

- `authorize({sessionId})` checks business inbox/outbox admission only. It must
  return `{allowed:true}` before provider dispatch.
- `tool({sessionId,toolCallId,name,arguments,messageId})` executes the frozen
  business capability. `messageId` is its already synced assistant anchor.
  Return `content`, `details`, `isError`, optional `terminate`. Tool content may
  include images. A durable host receipt precedes successful callback return.
- `event({sessionId,event})` is a **public notification**, not a durable model
  history ACK. Native reasoning, signatures, provider packets, and whole-history
  snapshots never return to Python. The awaited official turn-end boundary lets
  Python submit arrivals reentrantly after a complete tool batch.

Do not await `abort` or `close` from inside the event callback they are waiting
for; return the notification first. Tool cancellation is not proof an effect did
not happen. Unknown outcomes stay paused until a truthful durable receipt exists.
A failed callback creates a durable SDK failed result plus an unknown-outcome
marker; no automatic resend or additional model request follows.

## Prefix, recovery and durability

Only business tools are registered. Default read/bash/edit/write tools, skills,
project context, extensions, coding docs and workspace resource prompts are
excluded. The official `before_agent_start` forced-prompt override sends exactly
the configured system text, including the empty-string case, without Pi's cwd or
coding persona. No semantic Goal/NextSteps fields are required.

Automatic compaction/retry/cache warming are disabled. Narrow pinned-SDK guards
reject recovery `context_edit` and automatic compaction. Explicit compaction
uses `AgentSession.compact()`; a successful synced compaction entry alone changes
the visible projection. Canceled/failed/concurrent compaction preserves the old
prefix and admitted input. Every successful summary includes a bounded scoped
archive handle for public original evidence.

Maintenance preserves the exact visible tail after the last durable successful
main-turn boundary, including fresh text, image parts and reference versions.
Only the completed prefix reaches its summary task. With no completed turn it
does not compact; a failed or canceled run never advances that boundary. These
markers live only in the SDK archive and are not extra model messages.

Maintenance pressure uses the pinned SDK's `estimateTokens` for new messages,
anchored to valid provider usage (including cached input and the generated
response) when available. Image Base64 and opaque signatures are never counted
as ordinary text. Usage predating the latest compaction is invalidated even if
its assistant message remains in the retained tail. `state.contextTokens` and
`contextTokenSource` expose this advisory count; the host uses its token watermark,
not the legacy serialized-character watermark. This is not a hard capacity proof.

For an explicitly authorized system-prompt change, stop the sole group writer
and back up its data/config first. Under the same session-file lock, use the SDK
SessionManager to append a new `maibot.contract`, changing only `system`, and
update the group's prompt to match. Do not rewrite or delete old entries. Init
checks the latest contract; unrecorded system/tools/bindings/route drift still
fails. A rollback to the old prompt also requires an explicit contract update.

Pi normally leaves setup/custom-only entries in memory. The durability adapter
forces the SDK's own first-flush path, then `fsync`s the file and directory before
ACK. It does not build a second transcript. Any append/sync failure permanently
poisons that live session, because SDK memory may already have changed. No later
wire/tool dispatch is allowed. Restart strictly rejects malformed JSONL, missing
terminal newlines, duplicate entry IDs or invalid ancestry; do not silently drop
a torn record or automatically truncate a file.

A kernel `flock` helper holds the per-group writer lock. A second process fails
fast. Parent death closes the helper's stdin and releases ownership; unexpected
helper loss poisons the live session. Lock files must not be unlinked while the
service is running, which would split inode ownership. `close` waits for SDK
settlement before releasing the lock.

Provider route/thinking-level changes are explicit errors, not silent fallbacks.
Unknown context capacity remains honestly unknown (`contextWindow:0`); configured
strict admission is respected, while default audit mode is not a hard bound.
Configured output budgets are enforced at final provider wire construction. Task hard timeouts apply per provider response, not across business-tool execution. Error, canceled/timed-out, and length-incomplete model responses are not successful host turns.

Focus mode keeps its business readiness/debounce/wait checkpoints, but dispatches
each chat through its own Pi runner without shared model history or cross-chat
references. Its compatibility business timeline is not model authority and can
still grow with processed entries; retention of that scheduling checkpoint is a
separate maintenance concern, not model-context compaction.
