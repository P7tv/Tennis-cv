# Changelog

## 0.3.9 — 2026-07-29 (schema null-rule compliance + first handover package)

Client sent the full schema spec doc (`cv_schema_table_loeuf`) — audited
`build_loeuf_schema()` output against it field-by-field. 142/148 fields were
already present, but the doc's "never omit a key" rule was violated in
several places: keyframes (023-B) only emitted `{frame_index, detected}`
instead of the required `{frame_index, timestamp_ms, detected, joint}` (joint
= 13 named 2-D landmarks), the ball block (034-BL) fallback collapsed to a
single `{"available": false}` key instead of all 11 BL fields, `stroke_type_distribution`
(MT10) silently dropped zero-count stroke types, and the visualization block
(043-VZ) omitted keys entirely instead of nulling them when ball/racket data
was unavailable. `trophy_position` (B6, serve-only) and four serve-only BL
fields (`server_position`/`target_box_correct`/`serve_attempt_number`/`is_fault`)
were missing altogether.

Added `loeuf_cv/schema_fields.py` as a single source of truth for these key
lists (the duplication across `ball.py`/`metrics.py`/`schema_builder/builder.py`
was exactly what caused the drift), fixed all five holes in `schema_builder/builder.py`,
and added 3 tests that assert full key-set equality between the
`ball_traj`-present and `ball_traj=None` code paths (the case that was
silently broken). Deliberately left `stroke_specific` and `visibility_flag`
untouched — both are already compliant per the doc's own rules once verified
against it. Test suite: 65 → 68 passing.

Also assembled the first handover package matching the client's required
structure (`configs/`, `checkpoints/`, `dataset/{labels,metadata.csv}`,
`scripts/{train,predict,eval,preprocess}.py`, plus new `docs/FINETUNE_GUIDE.md`
and `docs/BENCHMARK.md`) — none of this existed before; the repo only had
its own internal layout. Rotated a hardcoded Roboflow API key out of
`scripts/train_yolo.py` into an env var while doing this (was committed in
git history — should be revoked on the Roboflow side).

## 0.3.8 — 2026-07-11 (SAM 2 vs classical CV, head-to-head on the actual bad clip — real win, with two caught bugs along the way)

Direct follow-up to 0.3.6/0.3.7: instead of patching `multi_person.py`
further, tested whether SAM 2 (0.3.5 prototype) actually solves the
repeated-position dropout bug on `805369423.086175.mp4` — the same clip
used to diagnose it.

**First attempt was wrong — caught before reporting it as a result.**
Ran SAM 2 on a 650-frame trimmed segment: got 100% coverage, looked like
a clean win. Checked the actual bounding boxes before trusting the
number and found all 650 were `(0, 0, 958, 538)` — the *entire frame*,
every single frame. SAM 2's video predictor loads all frames into
device memory at once; at 780 frames on this Mac's MPS it crashed
outright ("Invalid buffer size: 9.14 GiB"), and at 650 frames it didn't
crash — it silently produced a degenerate mask covering 100% of pixels
(confirmed on the very first prompted frame, before any propagation).
The "100% coverage" was MediaPipe running on a near-whole-frame crop and
coincidentally finding a plausible pose most of the time — not real
tracking. At the original validated size (150 frames, from 0.3.5) the
mask is correctly tight (bbox area ~5% of frame). So there's a real,
serious scalability ceiling for this environment: safe somewhere under
~300 frames (~10s), unreliable above that, unrelated to the tracking
quality question this test was supposed to answer.

**Also found**: the installed `sam2` PyPI package (`sam2==1.1.0`) is not
Meta's official release — `pip show sam2` reports
`Home-page: https://github.com/JinsuaFeito-dev/segment-anything-2`,
a third-party fork. It bundles Apache-2.0 license files copied from the
original, so redistribution looks compliant, but it's missing the
compiled `_C` extension (`cannot import name '_C' from 'sam2'` on every
run), which is the likely reason post-processing silently no-ops and
plausibly related to the memory behavior above. Before this goes near a
paid deliverable, should switch to installing from
`github.com/facebookresearch/sam2` directly.

**Redone properly**: extracted a 150-frame window (original frames
280-430) that actually covers the known-dead 325-361 zone, re-ran SAM 2
there, and this time sanity-checked the raw box sizes (33k-231k px²,
nowhere near the 515k px² full-frame area) before trusting anything.
Visually confirmed correct on 5 checkpoint frames including one inside
the dropout window — tight box on the right player, mid-swing included.

| | classical CV | SAM 2 |
|---|---|---|
| Coverage, frames 280-430 | 72.0% | **100%** |
| Mean core visibility | 0.592 | **0.818** |
| Known-dead window (325-361) | 0% ok (mean 0.00) | **100% ok (mean 0.87)** |

This is the first *honest, verified* side-by-side comparison in this
project between the two approaches on a clip with the real bug present
(action-plan item 2.4). SAM 2 fully resolves the specific dropout that
0.3.6/0.3.7 couldn't fix with three different classical-CV patches, on
the exact window that was dead before. Caveat: only tested on a 150-frame
window, not the full 72s clip, because of the memory ceiling above —
running SAM 2 on a full real clip on this machine needs either chunked
processing (propagate in overlapping windows, stitch) or a GPU machine
with more memory headroom.

## 0.3.7 — 2026-07-11 (tried a 3rd fix for the 0.3.6 dropout bug — hybrid frame-diff rescue, also reverted)

Follow-up to 0.3.6. Tried a more targeted fix: keep the original
whole-clip-frozen MOG2 model (best general accuracy of everything tested
so far) but add a fallback — when a currently-active track gets no
matched blob for a frame, search a margin around its last known box
using frame differencing against a few frames back (`_frame_diff_rescue`,
in `multi_person.py`) instead of relying on MOG2 alone. The idea: frame
diff isn't biased by long-term history the way MOG2 is, so it should
still catch small residual motion (weight shift, breathing, racket prep)
in a spot MOG2 has already learned to call "background."

**Result, measured honestly on the same bad clip
(`805369423.086175.mp4`)**: the mechanism works exactly as designed —
almost too well. The ball-feeder ("far" track, previously 2.5% coverage
because they stand still almost the entire clip) jumped to 92.4%
coverage. But the actual target player ("near" track) *regressed* from
92.7% to 76.5%, including a new 361-frame (12s) blackout at the very
start of the clip that didn't exist before. Ran with `--max-players 4`
to see the hidden clusters: found a 4th cluster (`other`, 76.3%
coverage, 1644 frames) with visibility stats nearly identical to the
"near" track — strong evidence the near player's own track got split
into two fragments that `select_candidate_tracks`'s same-person
clustering failed to stitch back together, and with `max_players=2` one
fragment lost out to the now much-stronger far/feeder track and got
dropped entirely. Likely cause: a frame-diff-rescued box can look
different enough (shape/position) from the track's MOG2-detected boxes
that the downstream clustering heuristics (tuned against MOG2's
detection characteristics) no longer recognize it as a continuation of
the same person.

Reverted again — `multi_person.py` is back to the 0.3.6 state (plain
frozen whole-clip MOG2, no rescue fallback). Kept the `_new_track`/
`_ingest_box` extraction as a harmless internal refactor (behaviorally
identical, just de-duplicated) and extended `scripts/diagnose_dropout.py`
to accept an optional role argument, for whoever picks this up next.

**Three attempts in, three different real regressions** — segment-based
warm-up (0.3.6), causal continuous adaptation (0.3.6), and now frame-diff
rescue all improve the specific known-bad windows but break something
else non-obviously, each in a different part of the pipeline (general
noise, a different clip, and track-clustering respectively). This is
no longer "hasn't found the right constant yet" — it's a sign that
patching detection alone isn't enough without also revisiting
`select_candidate_tracks`'s clustering assumptions, which were tuned
against MOG2's specific failure characteristics. Recommend the next
attempt either (a) fixes clustering to tolerate boxes from a
frame-diff-like source before retrying this approach, or (b) invests in
the SAM 2 path (0.3.5) instead, which has no background-subtraction
step and so doesn't have any bug in this family at all.

Test suite unaffected (62/62 pass).

## 0.3.6 — 2026-07-11 (root-caused a real dropout bug — attempted fixes regressed, reverted)

Investigated a client-reported issue: pose tracking "drops a lot" on some
real clips (`805369389.913558.mp4` good, `805369423.086175.mp4` bad —
both ball-feeding drill footage). Built `scripts/diagnose_dropout.py` to
dump per-frame core/wrist visibility + crop-box height + centroid speed
for the longest-tracked player, to test the initial hypothesis (dropout
during the swing/hit motion, from a client screenshot).

**Found two separate, distinct bugs — the swing hypothesis was only half
right:**

1. **Root cause of the big dropouts (not swing-related at all)**: both
   clips are ball-feeding drills — the player returns to the same court
   position and ready-stance pose between every single repetition.
   `_build_background_model()` in `multi_person.py` warms up MOG2 by
   scanning the *entire clip once* then freezes it (`learning_rate=0.0`)
   for the whole detection pass. Confirmed by pulling frames at every
   dropout window in both clips: every single one (4 in the bad clip,
   1 in the good clip, up to 5s long) is the player standing still at
   the *identical* spot/pose waiting for the next feed — never during a
   swing. Over a 70-120s clip the repeated static pose accumulates
   enough weight in MOG2's model for that specific position to be judged
   background — and since the frozen model is reused uniformly across
   the *whole* clip (including frames that happened before the model
   ever saw the repetition), this is also a real causality bug: early
   frames' detection is informed by frames from later in the clip.
   Coverage looked similar in both clips in aggregate (92.7% both) — the
   real difference the client felt is that the bad clip has *more*
   separate >0.5s blackouts (4 vs 1), because its drill cycle repeats
   more often.

2. **The actual swing-motion issue from the screenshot (real, but
   minor)**: wrist-landmark visibility specifically drops during full
   racket extension (mean 0.50-0.57, 33-42% of frames below 0.3) in
   *both* clips — confirmed visually against the reported frame. This is
   MediaPipe's own confidence dropping on an atypical/motion-blurred arm
   pose, not a tracking bug — core-body visibility stays fine at the
   same moments. Existing gap-fill/smoothing (`kinematic_fill`,
   `one_euro_filter`) already mitigates short instances of this; not
   separately addressed here.

**Attempted fixes for (1), both reverted after honest measurement:**
- *Segment-based warm-up* (rebuild MOG2 every `BG_SEGMENT_FRAMES`,
  tried 90 and 300): fixed the specific repeated-position windows but
  each segment restart is itself a cold model with no memory of "normal
  scene," causing far more scattered dropout elsewhere. Net: coverage
  92.7% → 62.5% (seg=90) / 75.2% (seg=300) on the bad clip. Rejected.
- *Causal continuous adaptation* (single pass, no freeze,
  `learning_rate=-1` instead of `0.0`, tried warmup=90 and warmup=200):
  did cut the worst blackouts on the bad clip (frames lost to >0.5s runs
  107→70 with warmup=90) but **regressed the good clip badly**
  (coverage 92.7%→63.4%, long-dropout-runs 1→18, frames lost 151→661).
  Net regression across the two-clip test set. Rejected.

Reverted `multi_person.py` to the original whole-clip-freeze
implementation (2 lines of diff net — `_build_background_model` and the
`learning_rate=0.0` call site are unchanged from before this entry).
**The root cause is confirmed and documented, but there is no fix yet
that is a net improvement** — both natural approaches (bound the
model's memory in time, or make it causal) trade the specific
repeated-position bug for worse general noise. Next ideas worth trying,
not yet attempted: hybrid MOG2 + local frame-differencing fallback so a
track isn't dropped just because MOG2 alone calls it background; or
adopt the SAM 2 prototype (0.3.5) for this class of clip, since
click-to-track has no background-subtraction step and so no version of
this bug at all.

Test suite unaffected (62/62 pass) — no net code change shipped, this
entry is a documented negative result plus a new diagnostic script
(`scripts/diagnose_dropout.py`) for whoever picks this up next.

## 0.3.5 — 2026-07-10 (SAM 2 click-to-track prototype — validated on real footage)

Track 2 from the action-plan doc: `webui/sam2_track.py` — click-to-track
using Meta's SAM 2 (Apache 2.0, safe for a paid deliverable — unlike
TennisCourtDetector's missing LICENSE from earlier research) as a
replacement for `multi_person.py`'s hand-built background-subtraction +
centroid-tracking + clustering stack, which took most of today
(0.3.0-0.3.2) to debug through a series of real, subtle bugs.

**Environment**: installed in an isolated venv (`webui/.venv-sam2/`),
not the project's main environment — SAM 2 pulls in specific torch/
torchvision versions that could conflict with MediaPipe's dependencies.
Ran into two real portability snags, both fixed:
- `decord` (SAM 2's default mp4 loader) has no prebuilt wheel for Apple
  Silicon — worked around by extracting frames to JPEGs via OpenCV
  first (`_extract_jpeg_frames`) instead, which has no such dependency
  and is more portable across platforms generally, not just this one.
- Device selection is automatic (`cuda` > `mps` > `cpu`) so the same code
  should pick up CUDA with no changes on the Windows GPU machine.

**Validated on real footage** (`805110616.390966.mp4`, 5s/149-frame clip,
tiny checkpoint, run on this machine's CPU/MPS — no CUDA available here):
single click at frame 10 on the near player's hip → SAM 2 propagated a
mask through the rest of the clip. Result: **139/149 frames (93%)
tracked**, mean visibility after feeding the resulting bounding boxes
into MediaPipe Pose was **0.77**. Checked correctness by drawing the
returned bounding box back onto the source frames at several points
across the clip (frame 10, 40, 80, 120, 148) — the box stayed tight and
correctly on the clicked player throughout, never drifting to the other
player (#52) or the ball-feeder standing nearby, despite 3-4 people
being visible simultaneously. This is the exact multi-person confusion
`multi_person.py` needed five separate bug-fix rounds to handle
reasonably — SAM 2 handles it correctly out of the box, with roughly
150 lines of integration code and zero custom clustering logic.

**Honest performance note**: ~1.5s/frame on this Mac's CPU/MPS (no
CUDA) — 149 frames took ~215s. This is expected to be much faster on
a real GPU (SAM 2 tiny typically runs at double-digit fps on CUDA); the
device-selection code needs no changes to pick that up automatically.

**Not yet done**: no click UI in the browser yet (coordinates are passed
as Python args for now); only tested on a 5-second clip (this machine's
CPU/MPS speed made a full-clip test impractical); multi-object tracking
(clicking more than one target player) is supported by SAM 2's API
(`obj_id`) but not yet exercised; no side-by-side coverage/identity-switch
comparison against `multi_person.py` yet (action-plan item 2.4).

No changes to `loeuf_cv/` or the test suite in this entry — this is
prototype code in `webui/`, isolated from the production pipeline until
validated further and a decision is made on adopting it.

## 0.3.4 — 2026-07-10 (One-Euro filter — and a real bug found in existing savgol path)

Track 1 item from the action-plan doc: added `one_euro_filter.py` as an
opt-in alternative to the existing Savitzky-Golay smoothing
(`PipelineConfig.smoothing_method = "savgol" | "one_euro"`, default
unchanged at `"savgol"` — no existing behavior changes unless explicitly
selected).

**What it is**: the 1€ filter (Casiez/Roussel/Vogel 2012) — a causal,
per-frame adaptive low-pass filter. Cutoff frequency scales with the
signal's own estimated velocity: low velocity (holding still) → low
cutoff → strong jitter reduction; high velocity (backswing→impact) →
higher cutoff → less lag. This is the actual property Savitzky-Golay
can't offer, since its window size is fixed for the whole clip regardless
of how fast the swing is at that moment. Also added confidence-weighting:
low-visibility frames get blended toward the filter's own running
estimate instead of trusting the (likely noisy/guessed) raw value
directly — "trust confident frames, guess through unconfident ones," as
specced.

**Real-clip validation** (not just synthetic): ran both methods on the
same real wrist trajectory (clip `805110616.390966.mp4`, near player,
after `kinematic_fill`). One-Euro cut jitter (frame-to-frame delta std)
from 0.00566 → 0.00124, about **78% reduction**.

**Found a real bug in the existing default path while comparing**:
Savitzky-Golay's jitter number came back *identical* to the raw,
unsmoothed signal — savgol wasn't just weak here, it applied **zero
smoothing at all**. Root cause: `smooth_timeseries()` only calls
`savgol_filter()` if `valid.sum() == T` — i.e. the *entire* clip must be
gap-free after short-gap interpolation, or smoothing is skipped
entirely for that landmark/axis. On this real clip's wrist landmark, 200
of 780 frames were NaN before gap-fill and 156 remained after (gaps
longer than `max_gap_fill_frames=5`) — so the all-or-nothing check fails
and **savgol silently no-ops**. Given how much of this project's real
footage has exactly this kind of occlusion pattern (documented
extensively in 0.2.x/0.3.x), this means the *default* smoothing path is
likely doing little to nothing on wrist/ankle landmarks in practice on
real clips, silently. This wasn't found or fixed as part of this change
(out of scope for the Track 1 ask) — flagging it clearly since it affects
the currently-default behavior, not just the new opt-in filter.
One-Euro doesn't share this failure mode: it's causal/per-frame, so gaps
just fall back to persistence (holds the last filtered value) rather
than disabling smoothing for the whole sequence — verified 0 NaN in the
One-Euro output on the same landmark where savgol left 156.

5 new tests (jitter reduction, beta-reduces-lag on a fast ramp,
confidence-weighting pulls a noisy low-confidence spike toward trend,
NaN persistence, default-unchanged guard) — 62/62 total pass.

## 0.3.3 — 2026-07-10 (internal inference test UI)

Added `webui/` — a Streamlit app for manually testing multi-person
tracking + pose extraction without running scripts one command at a
time. **Not a client deliverable** — the production pipeline still runs
headless via `scripts/`; this is purely a debugging/QA aid.

Phase 1 (MVP), built and smoke-tested:
- Upload a clip, set `max_players` / `model_complexity` / height /
  dominant side, run `extract_multi_person`.
- Track review table (coverage %, mean visibility per track) plus a
  rendered **skeleton overlay video** — the same visual-inspection
  technique that actually caught the wrong-identity and track-splitting
  bugs earlier today (numbers alone never would have).
- Per-track dropdown to label which track is which player (or "not a
  player" for ball-feeders/bystanders) — the click-to-select approach
  discussed as the practical fix for scenes with more than 2 people.

`webui/overlay.py` extracts the skeleton-drawing logic used throughout
today's debugging sessions into a reusable, testable function instead of
one-off inline scripts.

**Verified**: app starts cleanly (no console/server errors, checked via
browser preview); `render_overlay_video()` tested directly against real
clip data end-to-end (extract → kinematic_fill → render → labeled
correctly), output frame visually confirmed. Did not attempt a full
simulated browser file-upload (would require serving the local video
over HTTP first, for no additional confidence beyond what direct testing
of the same code path already gave — the file upload widget itself is
stock Streamlit, not custom code).

Not yet built (see `webui/README.md` roadmap): coverage-gap timeline
diagnostics, `court_click_tool.html` embed, full 17-layer JSON/metrics
viewer with confidence-flag highlighting.

## 0.3.2 — 2026-07-10 (track stitching: fixing the fix, twice)

Follow-up to a user request to validate 0.3.1's track stitching on the
*other* real clip (`805110616.192334.mp4`, the one with two people
genuinely standing close together at the net). This surfaced two more
real bugs in the clustering logic — each found and fixed with real
evidence, not guessed at:

**Bug A — transitive merge via an intermediate fragment (single-linkage).**
The original 0.3.1 implementation used union-find: union any two tracks
that pairwise satisfy "same person" (spatially close + low temporal
overlap). Running on `805110616.192334.mp4` merged the near and far
player (who are genuinely different people, previously validated as
correctly separated) into **one cluster** with 5 members. Root cause: a
third fragment was independently compatible with each of them, and
union-find chains transitively through it even though the near/far pair
is *not* directly compatible with each other (confirmed: 69% mutual
temporal overlap, well above the "different person" threshold). Fixed
by switching to **complete-linkage**: only add a track to a cluster if
it's compatible with *every* existing member, not just one.

**Bug B — complete-linkage over-splits on a single outlier member.**
Complete-linkage fixed Bug A, but broke `805110616.390966.mp4`: a
67-frame fragment that was clearly the near player (same region, x_mean
13.8px from the near cluster's mean, well under the separation
threshold) got stranded in its own cluster and mislabeled `far` —
confirmed by extracting the actual video frame at one of its active
timestamps: the "far" skeleton was sitting on the near player, not the
real far-court player. Root cause: complete-linkage requires
compatibility with *every* prior member, so a large cluster (already
absorbed several fragments spanning the whole clip) can reject a
genuinely-matching new fragment because of one uncooperative existing
member. Fixed by switching to **average-linkage**: compare each
candidate against the cluster's running aggregate (pooled centroids +
pooled tracked frames), not every individual member — tolerant of one
outlier, still correctly rejects genuinely-different people because the
aggregate's pooled frame-set still shows high overlap with them.

**Bug B's root cause, one level deeper.** Investigating *why* that
67-frame fragment existed as a separate track at all (rather than
already being merged at the blob-tracking stage) led to the real answer:
at several frames, the same physical player's silhouette was detected as
**two disconnected blobs** by background subtraction — e.g. frame 587:
one box `(284,174,24,39)` sitting directly above another box
`(294,209,34,107)`, i.e. head/shoulders separated from torso/legs by a
lighting/motion gap in the foreground mask. Each disconnected piece got
its own track ID, and because both pieces exist in the *same frame*, the
temporal-overlap heuristic ("tracked simultaneously → must be different
people") gets fooled — a single person's split-off limb looks exactly
like a second, smaller person standing behind them. Fixed at the source:
added `_merge_split_person_boxes()` to `detect_blobs()` — merges any two
blob boxes that are vertically stacked with a small gap and overlapping
x-ranges, before tracking even begins. This is a more fundamental fix
than patching the clustering logic, since it also reduces fragment count
directly (fewer, longer blob tracks → less clustering work to do at all).

**Final validated numbers on both real clips** (pose-success frame
coverage, `max_players=2`):

| clip | near | far | combined |
|---|---|---|---|
| `805110616.192334.mp4` (close pair) | 14% (97f) | 39% (271f) | **53%** (was ~21%/~1% before any of today's fixes) |
| `805110616.390966.mp4` | 74% (580f) | 14% (107f) | **88%** (was ~34%/~0%) |

Cross-checked clip1's near/far separation two ways: pose-success overlap
looked deceptively low (4%) because MediaPipe often fails on one of a
close pair per frame, but the reliable signal — raw blob-tracking
overlap — is 50.7%, confirming they're genuinely two simultaneously-
tracked people, not a wrongly-split one. Don't trust pose-success overlap
alone for this kind of check; use blob-level `tracked_frame_idxs`.

Added 4 new tests reproducing both bugs with synthetic data (transitive
merge via a bridging fragment, complete-linkage rejecting a valid
same-person fragment because of one outlier prior member, and 2 tests
for `_merge_split_person_boxes` — including one using the *exact* box
coordinates from the real frame-587 split) — 53 → 57 tests, all pass.

## 0.3.1 — 2026-07-10 (track stitching — recover fragmented coverage)

Follow-up to a user question: "why does the skeleton drop out for stretches
of the clip?" Traced it layer by layer with real numbers before touching
code (see below) — the answer turned out to be neither pose-model failure
nor occlusion, but a track-selection design flaw left over from the 0.3.0
fix earlier today.

**Diagnosis, ruled out in order:**
1. Pose detection failing on a tracked blob? No — only ~10% of blob-tracked
   frames had pose failure.
2. Player leaving frame / standing still long enough to be absorbed into
   the MOG2 background? No — visually confirmed the player was in frame,
   moving, at a moment the track had already ended.
3. **Actual cause**: the near player's blob was tracked correctly across
   almost the entire clip, but kept losing and re-acquiring across
   `MAX_TRACK_GAP_FRAMES` gaps (occlusion, fast motion) — fragmenting into
   **7 separate track IDs** covering frames spanning nearly the whole clip.
   Combined pose-success frames across all 7 fragments: 516/780 (66%). But
   `select_candidate_tracks` (from 0.3.0, same morning) kept only the
   single longest fragment (268 frames, 34%) and **discarded the other
   six outright** as "duplicates" — the duplicate-detection logic
   correctly recognized they were the same person, but the only action
   available was to drop them, not merge them.

**Fix**: redesigned `select_candidate_tracks` from "pick the single best
track per person" to "cluster fragments belonging to the same person via
union-find, then keep the highest-total-coverage clusters." Same
same-person criterion as 0.3.0 (spatially close + low temporal overlap),
now used to *group* via union-find instead of to *reject*. Return type
changed from `list[(tid, dict)]` to `list[list[(tid, dict)]]` (list of
clusters). `extract_multi_person` now stitches each cluster's fragments
into one continuous `PoseTimeseries`, using the fragment with the most
pose-success frames as the reported `track_id`/label.

**Validated on both real clips (not just predicted):**
- clip `805110616.390966.mp4` near player: 34% → **65%** coverage
  (268 → 516 frames), frame span widened from `[0,362]` to `[0,768]`
  (nearly the full clip) — matches the 66% predicted from manual
  fragment inspection almost exactly.
- clip `805110616.192334.mp4` near player: 21% → 36% coverage
  (145 → 253 frames).
- Re-rendered the skeleton-overlay video and confirmed visually: frame
  400, which previously showed **no one tracked at all** despite the
  player clearly being in frame and moving, now shows a correctly
  tracked skeleton under the same `track_id=4`.

Updated the 3 existing tests for the new return type and renamed
`test_select_candidate_tracks_skips_same_person_duplicate` →
`test_select_candidate_tracks_merges_same_person_fragments` (behavior
changed from reject to merge) — 53/53 tests still pass.

## 0.3.0 — 2026-07-10 (multi-person track selection: wrong-identity bug)

User caught this by watching a rendered skeleton-overlay video (see below):
the "far" role in `multi_person.py` was locked onto a bystander standing
near the fence, not the actual second player. Root-caused and fixed in two
rounds, each validated on both real clips before moving on:

**Bug 1 — persistence beats relevance.** `extract_multi_person` picked
tracks purely by `len(t["frames"])` (how many frames the track stayed
alive). A stationary bystander produces one easy, unbroken track; an
actively-moving rally player's track keeps fragmenting into many short
IDs (occlusion, crossing paths, distance). Confirmed on clip
`805110616.390966.mp4`: bystander track was 211 unbroken frames with
y-centroid std of 2.3px (essentially frozen) while the real far player was
present almost the whole clip but split into 13 fragments (y-std
15-52px, genuinely moving), none individually longer than the bystander.
Fix: added `MIN_MOVEMENT_STD_FRAC` — candidates whose y-centroid barely
varies are excluded before ranking.

**Bug 2 (found while fixing bug 1) — same person picked twice.** After
filtering stationary tracks, the next-best candidate by frame count
sometimes turned out to be the *same physical person* as an
already-selected track, just re-acquired under a new ID after a tracking
gap longer than `MAX_TRACK_GAP_FRAMES`. Confirmed: two tracks with
non-overlapping frame ranges (`[0,301]` then `[358,478]`, gap of 57
frames) and nearly identical spatial position got labeled "near" and
"far" — actually the same player. Fix: added `MIN_CANDIDATE_SEPARATION_FRAC`
— reject a candidate whose mean position is too close to an
already-kept one.

**That spatial-only fix broke a previously-validated case.** Re-checked
against clip `805110616.192334.mp4`, which has two real players standing
close together at the net (validated correct in an earlier session).
Their mean centroids were *also* nearly identical (same signature as
bug 2's duplicate) — but they're genuinely two different people tracked
*simultaneously* (69% frame overlap in raw blob tracking). Spatial
distance alone can't tell "same person, ID switched" (near-zero temporal
overlap) apart from "two different people standing close together"
(high temporal overlap). Fixed by requiring **both** spatial proximity
**and** low temporal overlap before treating a candidate as a duplicate.

**A further subtlety found during that fix**: the overlap check first
used pose-*successful* frames only, which under-counted overlap — when
two people stand close together, MediaPipe often successfully detects
only one of them per frame, alternating, so their "successful" frame
sets look artificially disjoint even when both blobs were tracked
simultaneously. Fixed by tracking a separate `tracked_frame_idxs` set
(every frame the blob was matched, regardless of pose success) and using
that for the overlap check instead.

**End state, validated on both real clips together** (not just
individually): clip1 correctly keeps both simultaneous close-together
players (`near`=track 0, `far`=track 3, matching the earlier validated
result); clip2 correctly excludes both the bystander and the
same-person duplicate, landing on the actual far player (`far`=track 47,
53 frames — down from the erroneous 211, but now the *right* 53 frames).
Refactored the selection logic into a standalone `select_candidate_tracks()`
function so it's unit-testable without a real video/MediaPipe. 3 new
tests lock in all three scenarios (reject stationary bystander, skip
same-person duplicate, keep two simultaneous close-together people) —
50 → 53 total tests.

Also rendered full clips with skeleton overlay (`near`=orange, `far`=cyan,
confidence-graded dot color) to `~/Downloads/` for visual QA — this is
what surfaced the original bug, and is now a recommended sanity-check
step before trusting any given clip's multi-person output.

## 0.2.9 — 2026-07-10 (kinematic gap-fill for occluded wrist/ankle)

Follow-up to measuring real wrist-visibility on the 2 client sample clips
(see 0.2.8-era investigation): core landmarks (shoulder/hip) were
95-100% visible, but wrist/hand visibility ranged 0.12-0.88 depending on
side/track, with low-visibility runs up to 65 frames long — often genuine
self-occlusion (arm swings behind the torso from the camera's viewpoint
during a stroke), confirmed by visually inspecting crops. Tried and
**rejected** two fixes first: (1) crop+zoom on just the arm/hand region —
made things *worse* (mean visibility 0.19→0.015) because a partial-body
crop violates the whole-body pose model's implicit assumption of seeing a
complete person; (2) MediaPipe Hands (a hand-specific detector, doesn't
need full-body context) on the same crops — only 9% detection rate (12/133
frames), because the underlying image quality (night, hand wrapped tight
around the racket grip) is genuinely hard even for a specialized tool.

New module `occlusion_fill.py`: motion-capture-style kinematic gap-fill.
For low-visibility runs on a "child" joint (wrist, ankle), uses the more
reliable "parent" joint (elbow, knee respectively) as an anchor + a
per-clip-calibrated limb length (median, robust to noise) + a cubic
spline on the parent→child unit direction vector (assumes the limb
rotates smoothly, not erratically) to reconstruct a position, then
**blends it 50/50 with plain linear interpolation** on the raw position.

**Honestly validated with held-out tests on real clip data** (hid a
window of *known-good* wrist positions, reconstructed them, measured
error against the real values — not just eyeballed):
- Neither method alone is reliable by itself: pure kinematic reconstruction
  beat pure linear interpolation in 3 of 5 tested gap sizes (10-50 frames)
  but was *worse* in the other 2 (worst case: linear 173mm mean error vs
  kinematic 205mm at a 50-frame gap). Blending the two eliminates the
  worst-case failures of either — the blend matched or beat the better of
  the two pure methods in every tested gap size, in both 3D world-landmark
  space and 2D image-normalized space (the space most metrics.py fields
  actually read).
- **But absolute error stays large**: even with the blend, reconstruction
  error is roughly 35-75% of the limb's own length (worse for longer
  gaps) — nowhere near the ~15% noise level that would make this
  trustworthy as a real measurement. This is a genuinely-better estimate
  than raw noisy MediaPipe output during occlusion, not a substitute for
  detection.

Given that, scoped conservatively: `MAX_KINEMATIC_GAP_FRAMES = 20` — only
fills gaps at or below this length (reasonable error), leaves longer gaps
untouched (existing `low_confidence`/`not_detectable` behavior). Critically,
**visibility scores are never modified** by this fill — CF1
(`pose_estimation_confidence`), CF3 (`partial_occlusion`), and the VF
`visibility_flag()` all keep seeing the original (low) visibility, so
downstream confidence flagging still correctly reports these frames as
estimated, not measured. Wired into `StrokePipeline.process_timeseries()`
right before `smooth_timeseries()`. 3 new tests (synthetic-data mechanics
only — the actual accuracy numbers above came from real-clip held-out
validation, not encoded as hard test assertions since real-clip data isn't
checked into the repo).

## 0.2.8 — 2026-07-09 (vanishing-point focal length — tested, mostly doesn't help here)

Followed up on the user's ask to research fully-automatic (no manual
clicking) camera calibration options. Researched current approaches
(sports-field-registration literature: PnLCalib, soccer-field CNN keypoint
methods, classical single-view metrology / vanishing-point calibration) and
implemented the one piece that's pure geometry — no training data, no
pretrained model, no licensing question:

New module `vanishing_point.py`: detects line segments (LSD) on the
court-surface region, estimates two vanishing points via two-round RANSAC
(one for the baseline/service-line direction, one for the sideline
direction — these are genuinely orthogonal in 3D since the court is a
rectangle), then recovers focal length via the closed-form single-view
metrology formula `f = sqrt(-(v1x-cx)(v2x-cx) - (v1y-cy)(v2y-cy))`
(Criminisi/Reid/Zisserman). Verified against exact synthetic geometry:
recovers focal length to <1e-3 px error when VPs are computed from clean
line endpoints.

**Honest result: this does NOT fix our actual problem.** Swept the
technique across camera pan angle (0° to 15°) and found the formula is
only numerically stable when the camera has substantial pan (yaw) offset
from the court centerline:

- At pan=0-2° — **the standard "camera centered behind the baseline" setup
  this whole project has recommended throughout** — the baseline/service-line
  vanishing point sits almost at infinity (the lines are nearly parallel in
  the image), so 1px of line-detection noise on those lines swings the
  recovered focal length by 30-65%. Completely unusable.
- At pan=10-15° (camera well off to the side of centerline) the same 1px
  noise only swings focal length by ~3-5% — usable, but this isn't the
  camera position we've told users to use.

Added a `_reliability_check()` bootstrap gate (perturbs inlier line pairs,
rejects if resulting focal length has >15% relative std) so the function
returns `None` rather than a confident-looking wrong number — it will
return `None` most of the time on this project's standard footage, and
that's intentional, not a bug. 3 new tests lock this in, including one that
explicitly asserts the gate rejects the centered-camera case.

**Conclusion**: kept as an available tool (useful if a client's camera
happens to be angled, or as a future refinement), but **not a fix for the
core problem** — the manual point-click tool
(`court_click_tool.html` + `search_camera_pose_from_points`, with the
depth-spread requirement from 0.2.7) remains the only currently-reliable
path to auto-height, regardless of camera framing. Fully-automatic
(no-click) calibration that works for our centered-behind-baseline setup
would need either a trained keypoint detector (see research notes below —
not yet attempted, licensing/domain-gap unresolved) or a different
geometric trick that doesn't degrade at pan≈0.

**Research notes on the deep-learning route** (not implemented, evaluation
only): [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
is a pretrained CNN (TrackNet-style, heatmap-based) detecting 14 tennis
court keypoints, claimed 0.96 precision on its own dataset. Two blockers
before it could be used: (1) no LICENSE file in the repo — using it in a
paid client deliverable without contacting the author is a legal risk;
(2) unknown domain transfer — its training data is presumably
broadcast-style (elevated, far) camera framing, not the close/ground-level
behind-baseline framing this project targets, so it would need empirical
testing (and likely fine-tuning) before trusting it. [PnLCalib](https://arxiv.org/abs/2404.08401)
(2024, soccer) validates that our `court_model.py` architecture (evidence
+ 3D-model optimization) is the right general pattern — the missing piece
is a more reliable perception/keypoint layer than classical Hough/Canny,
which is exactly what a trained keypoint model would provide if the two
blockers above were resolved.

## 0.2.7 — 2026-07-09 (same-depth click degeneracy guard)

Follow-up to a user question: "if we know the net's standard dimensions,
isn't clicking just the net corners (post left/right, base/top) enough?"
Tested it directly rather than answering from intuition — **it isn't, and
the failure mode is worse than just "noisy":**

- All 4 net-post points sit at the same depth (Z = 11.885m from baseline).
  Even with **zero click noise**, `search_camera_pose_from_points` recovers
  a *wrong* pose (height 2.16m vs true 1.3m, distance 11.56m vs true 6.0m,
  focal 1313px vs true 1000px) that still scores 0.95 against those same 4
  points — a genuine degenerate solution family, not search failure.
- With ~2px click noise: focal length std 210px (~21% of true value),
  distance std 3.8m (~63%), height std 0.43m (~33%) across 10 trials —
  unusable.
- Adding just 1 extra point at a different depth (e.g. service line center)
  helps some but not enough (focal std still ~164px, ~16%).
- Adding 2 extra points at a different depth *and* spread left-right (e.g.
  both near-baseline doubles corners) fixes it well: focal std 4.9px
  (~0.5%), distance std 0.03m, height std 0.02m.

Added `depth_spread_m()` + `MIN_DEPTH_SPREAD_M = 3.0` to `court_model.py`
as a reusable diagnostic (not a hard gate on the solver itself — a
low-confidence answer can still be better than none — but `solve_camera_pose.py`
now prints a prominent warning when clicked points don't span enough depth,
and `court_click_tool.html`'s on-page hint now explicitly warns against
clicking only same-depth points). Two new regression tests lock in this
finding (`test_same_depth_points_are_degenerate_even_without_noise`,
`test_depth_spread_m`).

**Practical guidance for anyone using the click tool**: always click at
least one, ideally two, points at a depth clearly different from your main
cluster (e.g. don't rely on the net alone — add baseline or service-line
corners too), spread left-right rather than clustered.

## 0.2.6 — 2026-07-09 (manual point-correspondence pose solve)

Extends `court_model.py` for the case edge-alignment search can't handle well:
camera zoomed in enough that court corners/lines don't fully fit the frame,
or automatic line detection is too unreliable to trust (night footage,
occlusion). Instead of requiring the full edge map, a human clicks whatever
known court features **are** visible — as few as 4 points, any mix of line
intersections, points along a visible line, or the net posts — and the
camera pose is solved directly from those correspondences.

New API: `PointCorrespondence` (image_xy ↔ world_xyz), `point_on_line()`
helper for picking a point along a known court line, `reprojection_error_px()`
/ `score_pose_against_points()`, and `search_camera_pose_from_points()`.
Internally refactored the random+multistart-refine search engine
(`_search_pose_generic`) to be shared between edge-based and point-based
scoring instead of duplicated.

Point correspondences give an exact, differentiable objective (unlike the
binary pixel-hit edge score), so after the usual random+refine search a final
`scipy.optimize.minimize` (L-BFGS-B) polish pass is run — this was necessary:
coordinate-wise hill-climbing alone plateaued around 6px RMSE (score capped
~0.15 under the `1/(1+rmse)` scoring), while adding the gradient polish
reaches sub-pixel/exact recovery on clean synthetic data.

**Honest results (synthetic, since no labeled real click data exists yet):**
- Noise-free clicks: 6 well-spread ground points (different depths/widths)
  are already enough to recover the true pose *exactly* — a net-post point
  isn't strictly necessary in this configuration. This is worth noting
  because the earlier assumption (elevated/non-coplanar points are required
  to break the focal-length↔distance ambiguity) turned out to be an
  overstatement for well-spread point sets; perspective foreshortening
  across depths already constrains focal length reasonably well.
- With ~2px Gaussian click noise (simulating realistic human clicking
  imprecision), recovered parameters stay close either way (focal length
  within ~1%, height within ~1.5cm, distance within ~6cm out of 8 trials) —
  but including a net-post point measurably tightens the noise sensitivity:
  focal_length_px std 9.2px → 5.0px, distance_m std 0.061 → 0.036 (roughly
  40-45% tighter, not a full fix of the ambiguity, but a real, reproducible
  improvement). Recommend including at least one net-post click when
  available.

**UI built and verified (same day)**: `labels/court_click_tool.html`
(no server needed, same pattern as `labeling_tool.html`) — load an
image or video, pick a feature from an 18-entry dropdown mirroring
`NAMED_POINTS`, click its location, export JSON. Companion
`scripts/solve_camera_pose.py` consumes that JSON, calls
`search_camera_pose_from_points`, prints the recovered `CameraPose`,
and (optionally) renders a green-line overlay + orange click-markers
onto the source frame for visual sanity-checking. Verified end-to-end
in an actual browser (not just unit tests): loaded a synthetic
reference frame, dispatched real click events on the canvas, confirmed
correct pixel-coordinate mapping and JSON export shape, then ran the
solver on the exported points and got back the exact injected camera
pose. Since the camera is static per court, this is meant to be solved
**once per camera setup** and the resulting `camera_pose.json` reused
across every clip from that camera — not re-solved per clip.

## 0.2.5 — 2026-07-09 (model-based court registration)

New module `court_model.py`: full pinhole camera model (`CameraPose` —
height/distance/tilt/pan/focal_length) + complete court line model (all
standard ITF lines: singles+doubles sidelines, both service lines, center
service line, net at correct height) + edge-alignment scoring + random+
multi-start-refine pose search. Answers "can we model the full 3D court
even when not all of it is visible" — **yes**: once *any* subset of
visible lines gives a reasonably-scoring camera pose, `project_court_lines`
predicts every line's image position including parts never actually
visible in frame (validated in tests — far baseline, 23.77m away,
still gets a coordinate even though it's off-screen for near-court framing).

This replaces the old `court_calibration.py` "must find exactly these 4
specific corners" approach with "use however many lines are visible as
evidence, score how well a hypothesized camera pose explains them all."

**Honest results:**
- Synthetic ground-truth tests pass (search recovers a pose whose
  projection matches the true pose's edge map, score >0.8) — but only
  after fixing a real bug: an early version let the search exploit
  near-invisible degenerate poses (almost the whole court projected
  off-frame, 1-2 leftover pixels happening to hit an edge = "perfect"
  score) — fixed with a minimum-evidence floor (≥40 sample points,
  ≥3 lines actually visible in frame) before a score counts at all.
- **Real footage: moderate, not solved.** Best score across multiple
  search restarts on the two real sample clips: 0.60-0.65 (partial
  line alignment — e.g. baseline roughly right, sidelines not well
  aligned) — better than nothing but not a clean full-court fit.
- **Focal-length ↔ distance ambiguity** (classic single-view calibration
  issue): different (height, distance, tilt, focal_length) combinations
  can produce near-identical 2D projections for a limited line set —
  meaning even a high edge-alignment score doesn't guarantee the
  recovered parameters are the *physically correct* ones, which matters
  for metric height estimation (not just visual overlay/prediction).
  Fixing the camera's actual focal length (e.g. from phone EXIF
  metadata, if available) would remove this degree of freedom and
  likely help substantially — not implemented yet.

Kept purely as an experimental/research module (not wired into
`auto_height_from_court` yet) — the underlying architecture is sound
and tested, but needs either richer real evidence (more visible lines,
known focal length) or labeled ground truth to become reliable enough
for production height estimation.

## 0.2.4 — 2026-07-09 (fixes attempt)

Investigated + fixed 3 concrete root causes of court-calibration instability
found in 0.2.3, but **overall feature is still not reliable enough for
production on real footage** — documenting honestly rather than continuing
to chase thresholds against 2 unlabeled sample clips with no ground truth.

Fixed (each individually verified):
1. **Sideline angle threshold** was tuned against synthetic-test geometry
   (>30°) but real baseline-camera framing shows sidelines at ~15-25° —
   widened `DIAGONAL_ANGLE_MIN_DEG` to 12°
2. **Background contamination**: light-tower truss structure was detected
   as a false sideline — now mask top ~35% of frame (fence/sky/tower zone)
   before line detection
3. **Corner sanity check**: near-parallel line misclassification caused
   line-intersection points to shoot off far outside the frame (observed
   x=-876px in a 960px-wide frame) → `find_court_corners` now rejects the
   whole corner set if any point falls implausibly outside frame bounds
   instead of silently producing a garbage homography
4. **Net-band search region**: was scanning the *entire* column from
   image-top to service-line looking for the longest dark run — on night
   footage the dark sky above the fence is a longer continuous dark run
   than the actual net band, so it was measuring sky height, not net
   height. Fixed: use inverse homography to predict where the net
   *should* appear (known depth 11.885m) and search a narrow window
   around that prediction instead
5. **Temporal-median reference frame** (`build_reference_frame`): reduces
   player-occlusion/lighting noise by taking the per-pixel median across
   many sampled frames before line detection (same idea as background
   subtraction in `multi_person.py`, inverted — extract the static
   background instead of the moving foreground)

**Net result**: each fix is real and verified in isolation, but chained
together the height estimates on the 2 real sample clips are still
implausible (rejected by the 100-220cm sanity range) or land on clearly
wrong values (299cm) when the sanity range doesn't happen to catch them.
Different frame subsets / sample counts converge on different, mutually
inconsistent homographies — meaning the classical Hough-line + angle-
heuristic corner detection is not accurate enough on this footage
(night lighting, fences, multi-person occlusion), not just noisy.

**Recommendation**: this needs either (a) real labeled court-corner
ground truth to properly validate/tune against, or (b) a more robust
corner-detection method (e.g. RANSAC quadrilateral fit enforcing the
known court aspect ratio, or a small trained court-keypoint model) —
not more manual threshold guessing against 2 unlabeled clips. Kept as
opt-in experimental (`auto_height_from_court=False` default); primary
path remains manual `--height-cm` or GAS-provided height.

## 0.2.3 — 2026-07-09 (later)

- **Court calibration** (`court_calibration.py`): homography from court-line
  corners (ITF standard dimensions) + weak-perspective height estimation
  (net height as vertical reference) — auto-derives `body_height_cm` +
  real-world cm scale without requiring manual height input. Core geometry
  math fully tested (homography round-trip, focal-length/height round-trip,
  synthetic-court corner detection all pass).
- `PipelineConfig.auto_height_from_court` (default `False`) wired into
  both `StrokePipeline.process_video()` and `SessionPipeline.process_video()`
- **Validated on real clips — honest result: not production-ready yet.**
  Perception layer (automatic court line/corner detection via classical CV)
  is unreliable on real night footage: estimated focal length varied ~2x
  across frames of the same static-camera clip, traced to line
  misclassification (background structures like light-tower trusses
  initially registered as false sidelines; fixed by masking the
  fence/sky region, but remaining sideline-angle detection is still
  inconsistent frame-to-frame). Kept as an opt-in experimental flag,
  not enabled by default — recommend GAS-provided height (per schema
  doc's Phase 2 note) or manual input as the primary path for now.

## 0.2.2 — 2026-07-09

- **Multi-person tracking** (`multi_person.py`): background-subtraction blob
  detection + centroid tracking + crop-zoom-then-pose, ให้ PoseTimeseries
  แยกต่อผู้เล่น (role near/far/other) — plug เข้า StrokePipeline/
  SessionPipeline เดิมได้ตรง ๆ ไม่ต้องแก้ metrics/keyframes
- Diagnosed on 2 real client sample clips (960x540, night, full-court
  framing, multi-person): whole-frame single-person detection failed on
  one clip (0.1% detection, remainder false-positive) despite camera
  position being correctly behind baseline — root cause was subject size
  in frame (~15% frame height), not lighting or camera angle/tilt.
  Multi-person module recovered a usable track (mean visibility 0.78,
  28% coverage) from the same footage without any new ML dependency
  (reuses existing MediaPipe Pose model on background-subtraction crops)
- `scripts/run_multi_person.py` CLI for diagnostics + Phase 1 handoff
- mediapipe pinned to 0.10.14 (0.10.35+ dropped the legacy
  `mp.solutions` API this pipeline depends on)

## 0.2.1 — 2026-07-03

- **Image normalization** (`image_norm.py`): gray-world white balance +
  exposure normalization + CLAHE เมื่อภาพมืด — ลดความต่างมือถือ
  แต่ละรุ่น/เลนส์ ก่อนเข้า pose model (VQ stats วัดจากเฟรมดิบเสมอ)
- **VQ edge guard**: ผู้เล่นชิดขอบเฟรม >25% ของคลิป →
  `player_near_edge` (enum เสนอเพิ่ม — กัน lens distortion ที่ขอบภาพ)
- D0.2 §1.1: device/lens/color normalization strategy

## 0.2.0 — 2026-07-02

Pipeline สมบูรณ์ตาม cv_schema_table_loeuf (17-layer schema) พร้อมรอ data

- **Schema**: rework ทั้งหมดเป็น 17 layers (MT/M/VQ/B/A + metrics 9 blocks
  + D/VF/CF/VZ + AGG/TRE/PAT/SUM) หน่วย cm/deg สัมบูรณ์
- **Calibration**: per-clip camera normalization — roll correction,
  per-frame scale (trunk anchor), rotation zero-reference, deviation
  envelope → รองรับคลิปมุมไม่เท่ากัน
- **fps**: input <60fps → resample เป็น 60fps (M4/M5)
- **Phase 2**: action spotting + hierarchical rule classifier
  (FH/BH/SV/VL/SL — ไม่คืน RS) + AGG/TRE/PAT/SUM ฝั่ง CV
- **Ball interface**: adapter + impact fusion + frame extractor
  (รอ detector จาก VM training)
- **Benchmark harness**: label CSV → accuracy ±1 เฟรม/ท่า, spotting
  recall/precision, agreement, accuracy-vs-deviation → BENCHMARK.md
- **Tooling**: batch runner (parallel+resume), labeling tool (browser),
  label spec + client guide
- Tests: 18 ผ่านทั้งหมด (synthetic landmarks — ไม่ต้องใช้วิดีโอ)

## 0.1.0 — 2026-07-02

- Prototype แรกตาม TOR v2 (schema draft, normalized units)
