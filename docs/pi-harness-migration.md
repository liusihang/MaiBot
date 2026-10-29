# Pi harness migration and verification boundaries

The main runtime is always the official Pi coding-agent harness. The legacy
`pi_agent_enabled` config key remains parse-compatible but no longer switches
between two frameworks. See [the wire and durability contract](../runtime/pi_agent/README.md).

## Ownership and call inventory

| Entry point | Owner of model execution/history |
| --- | --- |
| Every group/private chat, including Focus-selected chats | Independent persistent Pi `AgentSession` / `SessionManager`; no shared Focus transcript |
| Explicit main-thread compaction and delegated research | Official Pi sessions; child results contain public sources and scoped archive handles |
| Luna participation, ten-minute person facts, summaries, profiles, expression/jargon learning | Short Pi sessions through the typed service facade |
| Image/emoji understanding, WebUI diagnosis/replay/text learning/AI search, MCP sampling, plugin LLM APIs | Same Pi service dispatcher; native continuation uses scoped Node handles, not Python copies |
| Text/image embeddings and ASR | Explicit Node capability adapters, not fabricated chat context |
| Image generation | Node native-image adapters/official supported provider; permissioned `llm.generate_image` capability |

Python retains source repositories and business learner buffers, inbox delivery
receipts, tool permission checks, QQ outbox/idempotence, and public result
projections. These are not canonical model history. SDK assistant reasoning,
signatures, provider replay data, finalized calls and results remain in Node.

Public Pi response/tool events also feed the existing monitor timeline and the
shared model-usage/cache accounting services. Each provider response is a monitor
round with its actual route, usage and tool receipts. These observation records
do not reconstruct or modify authoritative SDK history or expose native thinking
and signatures. Auxiliary inference uses the same accounting implementation.

Delegated research defaults to the approved read-only tool set when `tools` is
omitted. The default `gpt_web_search` and `gpt_read_search_result` tools require
the exact host plugin ownership `local.gpt-web-search`; the child cannot obtain
`reply` or image-sending permissions through a matching tool name.
One-shot text/VLM and known single-turn context callers use `retain_replay=False`;
context APIs retain scoped handles by default for real multi-turn work. Release
retained results explicitly with `release_response` / `release_responses`. Reusing
a public-only result as native history is rejected. Auxiliary private SDK files
are deleted on nonretained completion/failure, explicit release, or settled close.

Expression/jargon collection now runs on admitted original source batches, not
on a Python model-history trim. Existing learner thresholds, bounded pending
buffers, concurrency gates and failed-batch retention still apply. Pending
learning buffers retain their prior in-memory restart semantics; they are not
claimed to be exactly-once durable training receipts. The ten-minute person-fact
scheduler remains independent. Luna can decline participation without losing the
source in Pi's durable session.

## Honest admission and failure semantics

Default `audit` permits unknown context capacity and explicitly makes **no hard
capacity guarantee**. Explicit `strict` rejects unknown proof or known excess
before sending. Non-chat operations without operation-specific capacity proof
reject strict admission; embedding dimensions are not token counts. No model
fallback, budget clamp, temperature/reasoning downgrade or unchanged automatic
network retry is used. Main error/abort/length-incomplete outcomes do not advance
successful-turn business watermarks.

High-water marks use SDK visible-size/reported-usage signals only to request
explicit SDK compaction. They are not a tokenizer. Automatic maintenance must
preserve the still-unprocessed source tail, including images, and only a synced
successful SDK compaction can replace old visible history. Original archive
entries remain available. Failed/aborted compaction does not prune content.

## Migration boundaries

- Old Python native codec/ledger/summary/compaction implementations were removed,
  not kept as an alternate model loop. Existing old-format Pi-adapter history is
  **not silently imported or fabricated as signed SDK history**. Keep old data
  for inspection; this harness uses its dedicated SDK session directory.
- Focus retains readiness, debounce and authorized wait/stop scheduling, but
  dispatches each real chat separately. No other group's/private chat's raw
  content becomes an implicit reference. Its compatibility business checkpoint
  can still grow; it is not authoritative model context.
- Emoji IDs are persisted by actual session in a separate SQLite business store.
  Do not prune this mapping when compacting model context. Legacy process-local
  IDs below `1e9` expire; no universal collision guarantee is claimed for unknown
  historic large IDs.
- `src/A_memorix` edits are a minimal temporary integration patch for upstream
  `A-Dawn/A_memorix`, branch `MaiBot_branch`; see its modification policy. They
  route inference to the host facade without inventing alternate memory logic.
- External ACP agents remain an explicit external-tool boundary. Pi does not
  control or attest a third-party process's internal provider calls.
- The production Gemini drawing plugin's complete source is not in this repo.
  The checked-in patch documents the Pi image capability contract but is **not
  proof of a fully migrated/deployed production drawing pipeline**. Its real
  source is required for that migration. The checked-in Qwen image-embedding
  plugin uses the permissioned Pi capability and is tested separately.
- No live-model quality/naturalness, live QQ delivery, production plugin install,
  credential changes, merge, or deployment is part of the offline verification.

## Coverage migration

The former tests coupled to Python's duplicate native ledger/codec and custom
compaction loop were replaced with real-SDK behavior tests, not mock SDK classes:

| Prior contract | Current tests |
| --- | --- |
| Native reasoning/signature/replay and unchanged wire prefix | `runtime/pi_agent/test/{harness-proof,provider,runtime,inference}.test.mjs` |
| Durable append ACK, restart/dedup, poison after storage failure, single writer | `runtime.test.mjs`, `stdio.test.mjs`, `sidecar.test.mjs` |
| Compaction failure/cancel/concurrent sources, reference visibility and archive scope | `runtime.test.mjs` and `pytests/maisaka/test_pi_harness_host.py` |
| Source conflicts, frozen tool binding, uncertain effects, lost callback ACK/media | `test_pi_business_store.py`, `test_pi_harness_host.py` |
| Child isolation/public evidence, parent-readable sources | `test_pi_subagents.py`, `test_pi_evidence.py`, `test_pi_harness_host.py` |
| All Python categories cannot access legacy SDK/model networking | `pytests/services/test_pi_harness_bridge.py`, entry-specific service/MCP/plugin/WebUI/A_mem tests |
| Focus isolation and participation without lost intake | `test_pi_focus_runtime.py`, migrated Focus/participation/interrupt/workspace tests |

Run `npm --prefix runtime/pi_agent test` and the relevant Python test suites.
Use Ruff **0.15.11** for whole-repo lint and the repository incremental format
checker against the PR base; do not mass-format existing baseline debt.
