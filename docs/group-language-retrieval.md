# Group expression and jargon retrieval

`src.services.group_expression_search.search_group_expressions(session_id, intent,
context="", limit=6)` retrieves optional language references, not a generated reply.
Before every utterance, the main Agent calls this tool with its intended meaning,
checks the returned meaning/situation and real evidence, and can adapt a fitting
expression **or use none**. The lookup is required before each utterance; using a
candidate remains optional. Empty results, unsuitable candidates, or query failure
leave the Agent free to express its original thought in its own words. This is not a
fact-memory search: references never authorize treating a joke as a personal fact.

## Sources and ranking

The catalog applies current-chat/share-group and approved/active status rules **before**
ranking. Every call reloads it and revalidates ID + content version after embedding; removed or disallowed rows cannot reappear merely because
an old vector remains cached. Records removed or edited while embedding was in flight
are dropped before returning; `catalog_changed` / `stale_records_removed` disclose
this race. Newly changed/added records wait for a subsequent query, with incomplete
coverage explicitly reported. Canonical `expression:<id>` / `jargon:<id>` identities stay
separate, including two senses of the same phrase. No phrase-only deduplication is used.

Each semantic document contains phrase (400 characters), meaning (600), situation (600),
and up to three real source/context examples (220 characters each); total is capped at
2,400 characters. Historical entries with no evidence do not get invented examples.
The embedder uses these descriptions to retrieve an intention without literal overlap.

1. Global lexical ranking: Okapi BM25 (`k1=1.2`, `b=0.75`, positive Robertson IDF).
   Structural semantic labels are excluded from lexical documents.
   Chinese runs use overlapping character bigrams; isolated CJK characters and
   casefolded alphabetic/numeric words are retained. No dependency or tokenizer LLM.
2. Independent semantic ranking: unit-normalized cosine similarity, restricted to the
   current approved catalog and the actual query's observed embedding space.
3. Reciprocal-rank fusion: `sum(1 / (60 + rank))`, ranks begin at 1. Raw BM25 and cosine
   scores are diagnostic only, never added together. Missing routes contribute nothing.

Zero-overlap lexical documents do not enter lexical ranking. Semantic ranking uses
finite cosine scores without a model-dependent hard threshold; each route contributes
at most its top 50. Even a low/negative absolute cosine is a relative rank, **not**
confidence or permission to use the phrase. Main Agent applicability checks and abstention
are required. No eligible catalog entries, or an unavailable semantic route with no
lexical hits, return an empty candidate list. Result limit is 1–6 (default 6).
Returned phrase/meaning/situation are capped at 400/1,000/1,000 characters,
source/related IDs at 20, and examples at three 600-character authentic prefixes.
Truncation is explicitly marked, never paraphrased. The serialized candidate list
also has a 12,000-character hard budget: only a ranked prefix is kept and
`coverage.results_truncated` discloses omitted results.

## Cost, persistence and truthful coverage

Uses the existing Pi `EmbeddingServiceClient` and successful-vector SQLite cache/leases
from `emoji_retrieval`, with a **separate** `group_language_vectors.sqlite3` database.
Namespace identity comes from actual provider/model/configuration/dimension fingerprint;
query/document route mismatches are never mixed. Edited content gets a new hash. Cache
sharing does not grant cross-group retrieval: the current catalog remains the allowlist.

A query makes one embedding request (8-second timeout) and fills at most eight missing
document embeddings within a separate 8-second fill budget. Existing cached documents
remain searchable. Missing-document scheduling interleaves BM25-relevant records with
deterministic hash-order breadth, not registration-order first-eight. Subsequent calls
skip successful content and advance the index. These are individual embedding requests,
not a claimed provider-side batch. There is no extra selector/generator LLM.

`coverage` reports records indexed/total, completeness, actual vector-space fingerprint,
RRF constant, null semantic threshold and mode:
- `hybrid`: all currently eligible records have vectors and both routes were evaluated;
- `partial_hybrid`: full-catalog lexical ranking plus semantic ranking of an indexed subset;
- `lexical_only`: no usable semantic route; `embedding_error` identifies failures by class
  (only safe error classes are logged; provider response bodies are not exposed);
- `empty`: no eligible catalog records.

A fully evaluated route may have zero qualifying hits. `hybrid` does not mean every
candidate matched both routes. Ranking diagnostics explicitly show each route's rank
or null. Partial coverage must not be presented as full-library semantic top-k.

For predictable full-catalog retrieval, an operator can explicitly run
`warm_group_expression_index(session_id, max_items=100, max_seconds=60)` until `complete`.
Each invocation is bounded (maximum 1,000 items/300 seconds), reuses durable progress and
cross-process leases, and performs one route probe. This API is not automatically run
as an unbounded hidden paid background job. New/edited records temporarily make coverage
partial again. Database and ranking work run off the asyncio event loop. Cancellation
propagates, and the reused lease helper releases in-flight claims safely.

## Offline verification

`PYTHONPATH=. python -m pytest pytests/maisaka/test_group_expression_search.py -q`
uses simulated vectors only: Chinese BM25, genuine completed full-index hybrid,
zero-literal-overlap retrieval including cosine 0.2, RRF vs raw-score addition, homonyms, live catalog removal,
cache/content/model identities, bounded cold fill, explicit warmup, unavailable/mismatched
routes, empty results, cancellation/lease recovery and bounded inputs/output. No live-model quality/cost claim is implied.

## Model-visible result and learning examples

The tool text contains only `表达风格 / 情景 / 例句` blocks for expressions and
`黑话 / 含义 / 例句` blocks for jargon, with a combined maximum of six records.
No fixed reminder or diagnostic JSON is appended. IDs, versions, ranking scores,
linked-definition metadata and coverage remain in the structured backend receipt.
Source examples are authentic source messages; neighbour context is used for
review/retrieval but is not presented as a usage example. Missing examples are
labelled `暂无例句`.

New learning saves a 600-character prefix and full-text hash alongside the source
reference, using existing expression provenance and jargon evidence JSON storage.
There is no extra LLM to invent examples and no schema migration. Retrieval checks
the same-chat source and fingerprint before using the saved example. Deleted or
edited sources do not expose stale snapshots. An old expression without provenance
can acquire an example when it is observed and approved again, reusing its ID.

## Learning quality

Joint extraction and evidence review select concrete expressive features worth
borrowing, rather than generic speech acts, grammar, numeric replies or facts.
Characteristic wording, rhythm, inversion, ellipsis, irony and joke continuations
can qualify, including familiar phrases. There is no minimum yield: a batch may
produce no expressions. Alternate extraction, single-item review and text-import
prompts use the same criteria in Chinese, English and Japanese.

Jargon classification now uses one context-grounded model call returning boolean
`is_jargon` and `no_info`, plus `meaning`. It recognizes special conventional senses
even when the model already knows them, and separates them from personal aliases,
official names and ordinary technical abbreviations. Different context-free and
contextual explanations are no longer treated as proof of jargon. A definite
non-jargon decision clears an obsolete meaning. Invalid classification types do
not finalize a record; insufficient evidence waits for the next observation
threshold. Existing 4/8/25/100 scheduling and hook names remain; the hook's
`comparison_result` argument now carries the direct classification result.
Candidates must occur in the top-level human source text, not merely a nearby
message or quote. Learning examples remain original sentences, never generated.
