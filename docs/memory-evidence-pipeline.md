# Evidence-first personal memory and derived views

This fork keeps the existing claim ledger and queues. It does not implement or claim knowledge of ChatGPT's proprietary memory internals.

## Person extraction

The existing batch includes source time and separately attributed reply context. Only the author's own text can support a claim. The model returns the existing fields plus `stability` (`stable`, `temporal`, `uncertain`) and optional `replaces` (an existing claim ID). Temporary states retain their time bounds in text and go to recent interactions; uncertain explicit plans never become stable identity. Old persisted extraction receipts without the new field remain readable.

Only explicit correction/change language requests bounded old evidence, from the same person and chat. This avoids repeatedly feeding every saved memory to ordinary extraction. The host resolves the returned target to a saved version and rejects invented IDs, foreign authors and manual records. A correction creates an evidence-backed replacement, then transactionally supersedes the original claim while preserving its evidence and transition history. A crash between the two steps leaves the journal pending; retry reuses extraction and ingestion receipts. A subsequent manual change or newer version blocks stale automatic corrections. Replaying a completed correction never restores a later-deleted claim.

The replacement ID is an optional semantic decision, not an instruction supplied by chat text. Source checks do not prove that every model paraphrase is semantically correct; quality still requires replay evaluation. No arbitrary expiry period or model-reported confidence is used as a substitute for evidence. This bounded automatic path is not a general deletion interface. Candidate retrieval uses explicit correction/change cues (including moves and switches); implicit or unusually phrased changes may remain separate observations rather than automatically superseding an old fact.

## Shared evidence

Summary windows carry deterministic source-window IDs and source-message IDs. Profile generation consumes existing classifications and only asks the model to classify remaining evidence. A summary optionally provides event spans that exactly partition its text. The importer validates and retains these spans; the event builder revalidates them against the stored paragraph. A single validated source window can produce several events without another model call. Rewritten summaries, missing metadata and multi-window grouping use the existing segmentation path; one paragraph is never assumed to mean one event. Cache signatures include the grouping metadata.

## Public references

- [ChatGPT Memory FAQ](https://help.openai.com/en/articles/8590148-memory-in-chatgpt): user-visible review, correction and removal, not a disclosed internal schema.
- [LangGraph memory concepts](https://docs.langchain.com/oss/python/concepts/memory): separate semantic facts, episodic context and derived profiles.
- [Mem0 update operation](https://docs.mem0.ai/core-concepts/memory-operations/update): explicit updates to stored memories; no assumption that add performs all correction operations.
- [Letta memory blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks): bounded editable context; adopted as a design principle rather than another agent service dependency.

Changes to the bundled A_Memorix implementation are intentionally maintained in this independent MaiBot fork, as explicitly requested by its owner. No upstream submission or live-data migration is part of this change.

Existing identical statements retain their recorded classification; a weaker later mention does not silently demote a verified or manual fact. This change does not bulk reclassify historical facts.

## Episode incremental generation

An implementation/model generation change alone does not enqueue a completed source. New evidence, corrections and deletions still advance its source revision. Automatic materialization preserves existing group boundaries and reuses unchanged results across generation versions; only new or changed groups use the current segmenter. Existing episodes remain queryable.

Completed groups, including empty results, are checkpointed to `metadata/episode_groups.sqlite3` before source publication. This derived cache is keyed by evidence data and generation, so interrupted work can resume without repeating successful requests. Final publication retains the existing source-revision/lease CAS contract.

Manual upgrades use `memory_episode_admin(action="preview_rebuild", sources=[...])` first. The response reports source/paragraph counts, a conservative model-request upper bound and a preview ID tied to the evidence snapshot and generation. Execution uses `action="rebuild"`, `preview_id` and `max_model_calls`; stale previews or insufficient budgets do not enqueue work. The budget covers the selected sources in one operation and is checked before each segmentation request. A failed/interrupted manual job is not resumed by the unlimited automatic worker.

The WebUI offers the same preview, budget and confirmation flow. The CLI defaults to preview; execution requires `--wait --max-model-calls N`. Matching completed groups can reduce actual calls below the displayed upper bound. This patch is a scoped MaiBot live-cost repair in A_Memorix internals; equivalent upstream ownership remains `A-Dawn/A_memorix:MaiBot_branch`.
