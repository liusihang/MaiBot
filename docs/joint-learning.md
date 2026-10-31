# Joint expression and jargon learning

The active Pi source path and the Focus / history-trimming compatibility entry both
feed one persistent per-chat queue. They no longer run the separate ten-message
extractors. Real user text is queued independently of the participation decision.

A batch becomes ready when its oldest unprocessed message has waited 30 minutes,
or 100 new eligible source messages have accumulated, whichever comes first.
A runtime-owned timer checks persisted work even if no new message arrives. Each
batch consumes up to 100 messages; a backlog of full batches drains without waiting
another 30 minutes. Sources are deduplicated by real session and message IDs, and
are not silently discarded at the former in-memory 100-message cap.

One `learner` request returns both expression and jargon arrays, with source IDs.
Disabled categories return empty arrays. The prompt allows up to 20 distinct
expressions and 30 jargon candidates rather than forcing a fixed 3–5 expressions.
Model output is validated before publication; quoted/media/tool text and bot output
are not new learning material. Existing per-category source blacklists still apply.

New expressions are normalized and checked for exact duplicates before review.
When expression self-reflection is enabled, one batch review receives each candidate's
actual source plus the preceding 10 and following 10 stored chat messages in database
arrival order. Overlapping messages are supplied once with source/window references.
At a history boundary, available counts and the incomplete status are explicit;
missing subsequent messages are never invented or awaited indefinitely. The review
prompt rejects meanings that require missing context. Ordinary colloquial language
is permitted; invented background and one-off facts are not generalized into style.

A SQLite journal in the instance data directory retains source receipts, extraction
results, approved candidates and per-category completion checkpoints. Retrying a
failed write reuses successful model results. Already-written expressions are not
counted again and remain eligible for vector-index repair. Repeated identical jargon
evidence does not increase frequency; valid evidence remains after inference.
Failures retry at most three attempts with a five-minute delay, then retain a failed
batch for diagnosis while permitting subsequent work. This is not an automatic
operator-facing repair UI or a promise of distributed multi-process execution.

Plugin extraction / write hooks remain available. A hook may reject a reviewed
expression, but changing its reviewed meaning requires another review and therefore
fails the current write rather than silently publishing unreviewed content.

Legacy learner APIs remain for explicit callers, but ordinary Pi / Focus ingress
uses the joint coordinator. The separate jargon meaning-inference stages are not
redesigned by this change. No model-quality or live-cost claim is made by the offline
tests; those need authorized real-chat replay evaluation.
