# Emoji semantic retrieval

The built-in tool flow is now `query_emoji(query)` → at most three candidates →
the main Agent selects `send_emoji(candidate_index)` or `reply.attach_emoji`.
No selector LLM, random sampling, or random-first error fallback runs on this path.
An empty result is valid; the Agent can also decline all positive-similarity hits.

The query compares text embeddings of the query with the existing registered
emoji descriptions/tags, through `EmbeddingServiceClient` and Pi. It does **not**
compare an arbitrary text vector against the image-memory index: cross-modal
vector compatibility is not established by equal dimensions or a similar model name.
Only current registered emojis with nonempty descriptions and available files are
eligible. Images in general image memory are not part of this search.

Queries after registration or a description/model change generate missing
text vectors, bounded to eight descriptions and an eight-second indexing budget per
query. Results explicitly report indexed/total counts and whether the index is complete;
a later query continues the unfinished build. SQLite leases prevent concurrent queries
from embedding the same description; cancellation releases a lease and preserves
already completed vectors. Leases expire after a process crash. Successful vectors persist under `data/pi_harness/emoji_description_vectors.sqlite3`
and survive restart; unchanged descriptions are not re-embedded. The observed
model/provider/configuration fingerprint separates vector spaces. A mixed route
within one query fails explicitly instead of comparing incompatible vectors. A large library requires multiple cold queries or explicit warmup (below); no claim of measured online savings is made.

Candidate numbers use the existing durable, session-scoped emoji resource store.
They survive restart, cannot be reused in another chat, and are revalidated against
available registered resources at send time. Missing/unknown IDs fail without a
new selection or send. Hooks can veto a send but cannot silently replace an explicitly
selected emoji. Existing send receipts and destination checks remain in place.

`send_emoji` without a candidate now returns an actionable failure asking for a query.
`show_emoji_list` remains importable for old replay/tests but is not exposed in the
active catalog. The legacy `use_new_send_logic` switch no longer chooses a random
selector: both settings use the semantic query and explicit send flow. No live
configuration, credentials, deployed instance, or paid model invocation is changed
by this code migration.

## Explicit full-library warmup

Cold chat results are the best three **within the indexed subset**, not global
nearest neighbors. Chat queries alone advance by at most eight new descriptions;
there is no automatic background full-library build. Until warmed, earlier registered
resources can therefore be overrepresented. For a large library, an operator can
explicitly invoke the supported async maintenance callable below in the initialized
bot runtime (or a maintenance integration with the same instance configuration and Pi
service). This is a Python API, not a standalone shell command or an Agent tool:

```python
from src.emoji_system.emoji_manager import emoji_manager
from src.services.emoji_retrieval import warm_registered_emojis

progress = await warm_registered_emojis(
    list(emoji_manager.emojis),
    session_id=existing_chat_session.session_id,
    max_items=1000,
    max_seconds=300,
)
print(progress.indexed, progress.total, progress.complete)
```

Use an already resolved real chat session; do not invent a session ID. Warmup makes
paid embedding calls (one observed-route probe plus one per missing description),
not generation/selector calls. It shares the same cache and singleflight leases as
queries. Repeat the bounded invocation if `complete` is false; completed entries
are skipped, not restarted from the first eight. Cancellation preserves completed
work. A timed-out description stops that invocation; fix the provider problem and
resume. Concurrent owners may leave progress incomplete until their work completes.
Rerun after new registrations, description edits, or embedding route changes. No
warmup or live model request was executed as part of this code change.
