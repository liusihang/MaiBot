# Explicit memory query evidence contract

`query_memory` now keeps requested person, time, and real chat-session constraints.
A person-filter miss is `no_match`, not permission to return other people's hits or
repeat an unscoped search. A missing real session fails before searching. Requests
with time bounds use `hybrid` (or `time` without a query) when the requested `search`
or `aggregate` route would otherwise include unbounded semantic results. Receipts
show both requested and effective scopes/modes; no time or person bound is removed.
Configured shared-chat and global-source visibility remains governed by A_Memorix.

The retrieval minimum (`min_results`) is **unchanged** for recall callers. The
threshold stage attaches query-local `retrieval_evidence` metadata to copied
results, distinguishing threshold-passing hits from minimum-count backfill. Fact
projection/current-state refresh preserves this metadata. Explicit `query_memory`
does not put marked backfill into answer evidence: it returns `no_match` when only
such candidates remain. Stored JSON retains complete scope/evidence-qualified
`hits`; `excluded_candidates` contains only rejected IDs and reasons, **not their
text**. A later page or keyword snippet therefore cannot detach rejected text
from its label and misrepresent it as evidence. This is a relevance gate, not a factual-truth verifier. Unmarked
time/episode and smart-path results retain their existing semantics; the change
does not claim to solve every unknown name, attribution, or relevance error.
Legacy exact person-ID mentions remain searchable, but a mention is not proof
that every statement in that memory belongs to the mentioned person.

The tool stores complete scope/evidence-qualified hit content and source metadata in the existing
session-scoped, in-memory `ToolResultStore`. Its normal output is at most 2,400
characters, the replyer reference at most 2,200, and the structured receipt at most
8,000 serialized characters, with at most five preview hits. The receipt includes
`result_id`, `reader_tool=read_tool_result`, hit counts, and truncation status;
provenance IDs/URIs are kept whole or explicitly omitted with `omitted_fields`,
never shortened into a different apparent identifier. JSON escaping counts
against structured preview budgets.
`read_tool_result` can page/search that JSON in the same session. This is not a
lossless dump of the full raw retrieval candidate set. Existing
store eviction/restart semantics apply. Filtered/failed responses never store or
forward accompanying hits. No-match responses supply no replyer memory reference
and enqueue no feedback task. Feedback for previews retains the visible hashes
and sources; raw candidates are not represented as adopted answer evidence.

## Local A_Memorix compatibility exception

The small edits to `core/retrieval/threshold.py` and
`core/runtime/services/search_hit_processing_service.py` are a temporary local
compatibility fix under `src/A_memorix/MODIFICATION_POLICY.md`: MaiBot needs a
fail-closed person filter and a preserved backfill/evidence distinction before
its explicit tool can safely present scoped evidence. No global threshold,
recall count, retrieval algorithm, or stored-memory schema is changed. Equivalent
changes should be reconciled with upstream `A-Dawn/A_memorix` `MaiBot_branch`
before the next subtree sync. This work does not modify or publish upstream.

Regression coverage includes strict person filtering, low-score real retrieval
results through projection and host coercion into explicit tool receipts,
unbroadened time/person misses, filtered/error payloads, bounded large results,
raw source preservation, and cross-session read isolation. Tests use mocked
search/runtime services and temporary instance directories; no bot, model, or QQ
endpoint is called.
