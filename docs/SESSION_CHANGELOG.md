# What's been done on this project — a running record

This file exists so you can see, in one place, everything that has changed
across our sessions on this ALPR University Gate system: what was broken,
why, what was changed, and how each fix was verified. It is written in the
order the work happened.

Current status as of this writing: **812 tests passing, 3 pre-existing
failures** (unrelated to any of this work — see the end of this document),
both Docker services (`alpr-dashboard`, `alpr-api`) up and healthy, GPU
verified reachable inside the container.

---

## 1. The dashboard was lagging / hanging on every page load

**Symptom:** the Streamlit dashboard felt stuck — slow to load, slow to
process, and the `alpr-api` container was crash-looping.

**Root cause:** the running containers were bind-mounting a checkout at
`/home/ansh/ansh_2026/alpr-university-gate/...` that did not exist on this
machine. Docker had silently created empty, **root-owned** placeholder
directories in its place. The container ran as your user (`1005:125`), so
it could not write to those root-owned folders — every attempt to open the
log file (`RotatingFileHandler` in `src/utils/logger.py`) raised
`PermissionError`, which crashed the Streamlit script on **every single
page render**. The health check still reported "healthy" because it only
pings `/_stcore/health`, which never runs your actual code.

A second, unrelated problem was found at the same time: `.env` held a
stale `GID=125` (your docker group) instead of your real GID `1005`, and
the running image was months out of date relative to the code in this
checkout — missing `src/runtime/` entirely and carrying stale copies of
the detectors and OCR engines.

**What was fixed:**
- Recreated the Docker containers so bind mounts resolve to this actual
  checkout (`docker compose down` + rebuild + `up`).
- Rebuilt the image from the current source (heavy layers — PyTorch,
  TensorRT, Paddle — stayed cached; only the code layer rebuilt, seconds
  not 20 minutes).
- Corrected `.env` to the real UID/GID.

**Verified:** `PermissionError`/`Uncaught app exception` count in logs
dropped from "every page load" to 0. Both containers came up healthy and
stayed that way. Full import chain (`logger` → `config` → `camera manager`
→ `database` → `runtime.device`) completes in 0.2–0.4s with no errors.

---

## 2. Dashboard performance: caching, refresh loop, redundant disk reads

Separate from the crash above, four real inefficiencies were found and fixed
in `src/dashboard/trajectory_app.py`:

1. **Full-script refresh loop.** The page used to do
   `time.sleep(2.0); st.rerun()` at the end of every run while a session was
   live — re-executing the *entire* 1,800-line script every 2 seconds,
   including the sidebar, every database query, and all three tabs (Home,
   Operations, Trajectory) regardless of which one was open. Worse, the
   `sleep` blocked the script thread, so a click during that window wasn't
   acted on until the nap finished — that delay, not the rendering, is what
   made the page feel unresponsive.

2. **No caching at all.** Every rerun re-queried the database from scratch
   for the session list, blacklist sightings, heatmap counts, and
   statistics.

3. **All three tabs rendered on every run**, hidden or not, because
   `st.tabs` doesn't lazily render its bodies.

4. **Live camera frames read as full JPEGs from disk on every rerun**, even
   when only being used to check "does a frame exist yet" for a caption.

**What was fixed:**
- Replaced the sleep-and-rerun loop with `st.experimental_fragment` panels
  for the four things that actually move during a live session (header,
  camera wall, processing status, statistics) — each on its own
  `run_every=2.0` timer, independent of the rest of the page.
- Added `@st.cache_data(ttl=2.0)` wrappers around six database read
  functions, with explicit cache-clearing on the one write path (deleting a
  session) so an operator's own edit is never shown stale.
- Added a cheap existence check (`_has_live_frame`, seeks the last 2 bytes)
  used whenever the panel only needs a yes/no, instead of reading the whole
  file.

**Verified:** benchmarked identical 5-rerun sequences before/after —
database query count dropped from **30 to 5** across 5 reruns. Full
dashboard smoke-test suite (58 tests) passed before and after with the same
single pre-existing failure (see below), confirming no behavioural
regression.

---

## 3. A bug I introduced: `RuntimeError: Could not find fragment with id ...`

**Symptom:** clicking "Start processing" (or more precisely, the moment a
processing session ended) crashed the page with this exact error.

**Root cause:** this was a direct consequence of fix #2 above. Streamlit
identifies a fragment by `md5(module + qualname + container path)`, and
only registers that ID during a *full* script run in which the fragment is
actually invoked. The code called the fragment version of each panel while
a session was live, but a plain, undecorated render function once it went
idle. The instant a session finished, the next full run drew the page
*without* those fragments — so a refresh timer already ticking in the
browser arrived for an ID that no longer existed, and Streamlit killed the
run.

**What was fixed:** every live panel now has two forms — one with a
`run_every` timer, one without — both built by decorating the *same*
underlying function, so they hash to the *identical* fragment ID. The page
can switch between live and static freely with no orphaned timer ever
possible.

**Verified:** confirmed by hash inspection that all four panel pairs (`_auto`
vs `_static`) produce identical IDs. Added 5 new tests
(`TestLiveSession`, `TestLivePanelFragmentIds`) that exercise the
previously-untested "session is running" render path and assert both forms
of each panel share one identity — pinning this exact invariant so it can't
silently regress. Drove a real 4-camera session through the dashboard code
path end-to-end: completed cleanly, 4 detections per camera, no exception.

---

## 4. GPU silently lost by a long-running container

**Symptom:** a processing session failed instantly with
`ModelUnavailableError: detection.device=0 asks for CUDA, but no CUDA
device is available`, even though `nvidia-smi` on the host showed the GPU
working fine.

**Root cause:** a container that has been running for hours can lose GPU
access underneath itself — a driver update or a `systemctl daemon-reload`
resets the cgroup device allowlist, and the device nodes remain visible
inside the container but become unusable (`cuInit` fails). Nothing
announces this; it only surfaces the next time something tries to actually
use CUDA, which can be hours after whatever broke it. The dashboard
container had been running since before this happened.

**What was fixed:** added a cheap (~0.15s) direct probe —
`ctypes.CDLL('libcuda.so.1').cuInit(0)` — to the project's launcher script
(see item #5). `./run up` and `./run restart` now verify GPU reachability
after starting and **automatically recreate the container** if it's been
lost, and `./run doctor` reports this state explicitly (the host having a
GPU says nothing about a long-running container still having one).

**Verified:** confirmed the probe correctly reports `False` when the GPU is
genuinely unreachable and `True` once recreated. Ran a full real
multi-camera session end-to-end after the fix through the exact failing
code path — completed with `STATE: completed`, 4 cameras × 4 detections
each, no error.

---

## 5. Running the project was inconsistent and error-prone every time

**Symptom:** "every time I open the folder and run the project I face
issues" — stale `.env`, containers pointing at the wrong checkout,
forgetting `git lfs pull`, forgetting to rebuild after a code change, not
knowing whether to use Docker or the native `alpr/` virtualenv.

**What was built:** a single entry-point script, `./run`, covering the
whole workflow in one place, in both Docker and native-virtualenv form:

| Command | What it does |
|---|---|
| `./run up` | Start dashboard + API, waits until they actually answer before printing URLs |
| `./run down` / `./run restart` | Stop / restart (restart picks up code edits — no rebuild needed) |
| `./run video <file>` | Run the full pipeline over a video file, no browser |
| `./run rtsp <url>` | Run the full pipeline over a live camera |
| `./run test [args]` | Run the test suite |
| `./run doctor` | Diagnose the machine and this checkout; changes nothing |
| `./run status` / `./run logs` / `./run shell` / `./run build` | As named |
| `--native` flag, or `ALPR_RUNTIME=native` | Use the `alpr/` virtualenv instead of Docker |

Built-in guards, each closing a specific failure that actually happened on
this machine:
- **Foreign-container eviction** — detects and replaces any container whose
  bind mounts point outside this checkout (this is exactly what caused
  problem #1). Tested by deliberately planting a container with mounts at
  the old `/home/ansh/...` path; `./run doctor` flagged it and `./run up`
  evicted and recovered correctly.
- **`.env` auto-sync** — rewritten to the real UID/GID on every command, so
  it can never drift.
- **Git LFS check** — detects a plate-detector weight file that's still an
  LFS pointer instead of the real model.
- **Stale-image detection** — rebuilds automatically when
  `requirements.txt`/`Dockerfile` are newer than the built image.
- **GPU verification** — see #4.

Also changed:
- `docker-compose.yml` now bind-mounts `src/`, `scripts/`, `main.py`,
  and the test fixtures (`tests/`, `training/`, `ALPR.mp4`,
  `mycarplate.mp4`) that `.dockerignore` deliberately excludes from the
  image to keep it small. This means **a code edit now takes effect on
  `./run restart` with no rebuild step**, and `./run test` inside the
  container can actually find the files it needs.
- `docker/start.sh` now delegates to `./run up` so old habits/bookmarks
  still work.
- `pytest.ini` given its own cache directory (`logs/.pytest_cache`) since
  `/app` is root-owned inside the image and was spamming every test run
  with permission warnings.
- `README.md` quick-start section rewritten to match the new workflow.

**Verified:** ran a full clean-slate test — deleted `.env`, removed all
containers, ran `./run up` — dashboard and API came up healthy in **2.7
seconds**. Ran the full pipeline through both `./run video` (Docker) and
`./run --native video` (virtualenv) — both completed cleanly with identical
plate detections. Test count went from 713 passed/8 failed/2 collection
errors (before, missing fixtures) to 807 passed/3 failed (after).

---

## 6. Vehicle profile images permanently "lost" after a session delete

**Symptom:** a vehicle (`DL2CAT4762`) showed "No image stored" in search
results, even though its crop image genuinely existed on disk.

**Root cause:** a vehicle's summary profile keeps pointers to the images
from its single best (highest-confidence) detection. If that detection's
image files are later deleted — e.g. the crops from an old session lost
during the broken-mount period in problem #1 — the profile kept the
now-dead file *paths* forever. The code only checked "is this field set to
some string?", never "does the file this string points at actually exist?"
— so no later, real detection could ever replace the dead reference.

**What was fixed:** in `src/database/vehicle_profiles.py`, image selection
now checks file existence (`_image_exists()`), not just whether a path
string is present. A detection with real files on disk now replaces a
profile's reference to files that no longer exist, either on the next
sighting or on a profile rebuild.

**Verified:** added 2 regression tests
(`test_a_deleted_picture_is_replaced_by_one_that_still_exists`,
`test_a_rebuild_also_heals_a_dead_picture`) and confirmed both **fail**
against the original code and **pass** against the fix — proving they
actually catch the bug rather than just describing it. Ran
`rebuild_all_profiles()` against your real database: all 4 stored vehicles
went from partially-broken image references to fully resolving
(`usable=4/4` for every plate). Also had to fix 3 *existing* tests that
asserted on synthetic filenames that were never real files on disk (see
note at the bottom).

---

## 7. Recorded video playing at ~2x speed on the camera wall

**Symptom:** during a live processing session, the camera feed panels
played uploaded videos noticeably faster than real speed.

**Root cause:** nothing paced frame reads for recorded files — the
pipeline read frames as fast as the GPU could process them. On this
hardware that's roughly 60fps against 30fps source footage, i.e. exactly
2x. The detections themselves were unaffected; only the preview picture
was running at the wrong speed.

**What was fixed:** `src/capture/frame_capture.py` now holds recorded-file
playback to the clip's own frame rate via a new `_pace_to_realtime()`
method, gated by a new config flag `video.realtime_playback` (default
`true`). It only ever *slows down*, never skips frames — if processing is
genuinely slower than the footage, nothing changes, so no detections are
sacrificed to chase a clock. Waits are done against the existing stop
event, not a plain `sleep`, so pressing STOP mid-playback is still
immediate. Live RTSP sources are untouched (a real camera is already
paced by definition).

**Verified:** measured directly —
`realtime_playback=False`: 320 frames read in 0.31s (1026fps).
`realtime_playback=True`: 320 frames read in 10.66s (30.0fps, exactly
matching the source). Confirmed STOP still exits within 1ms even while a
pacing wait is in progress. Full pipeline run through `./run video`
matched real clip duration (~10.7s clip finished in ~12.8s including model
load).

---

## 8. Vehicle body-colour misclassified on tall vehicles

**Symptom:** a search for "white vehicles" returned only 2 results when 3
white vehicles had actually been seen — a white Maruti Omni van
(`DL7CD5017`) was classified `Silver` instead of `White` in 10 of 11
detections.

**Root cause:** `src/classification/vehicle_color.py` decides achromatic
colour (white/silver/gray/black) from the median brightness of a fixed
sampled region — 35%–80% of the crop's height, chosen specifically to sit
below the windscreen/rear-window band **on a car**. On a tall van, that
same fixed band lands squarely on the dark rear window instead, and the
window's darkness drags the median brightness down. Measured on the actual
crop: the van scored **179** against a **White** threshold of **≥180** —
it lost the classification by one brightness point, entirely because of
where the window happened to sit in the frame, not the paint colour.

**What was fixed:** rather than assume a fixed band avoids the glass, the
classifier now identifies which *rows* of the sampled region are actually
bodywork versus glass/trim, using row uniformity: a painted panel row is
one colour all the way across (low quartile spread); a glass row carries
reflections of trees/sky/road (high spread); a trim/chrome row is half
dark paint and half blown-out highlight (also high spread). Only uniform
rows contribute to the brightness measurement used for the
white/silver/gray/black decision. New config knob:
`vehicle_color.panel_row_spread_max` (default `60`, `0` restores old
behaviour).

This required two iterations — the first attempt (a simple two-cluster
brightness split) broke an existing test
(`test_chrome_and_glare_do_not_turn_a_black_car_silver`) by re-introducing
exactly the failure mode that test was written to prevent: chrome/headlight
highlights on a dark car falsely reading as a bright cluster. The
uniformity check specifically distinguishes a genuinely bright *panel* from
a bright *pixel* within an otherwise dark row, which is what fixes both
cases at once.

**Verified:**
- Real crop: van goes `Silver → White`, other 3 stored vehicles unchanged.
- All 28 existing colour-classification tests still pass, including the
  black-car/chrome-glare guard.
- Cross-checked against 184 real vehicle crops extracted from a *second*,
  unrelated video (`ALPR.mp4`): 176 unchanged (95.7%), 6 correctly
  corrected from `Gray → Black` on genuinely dark vehicles (a black Volvo
  SUV, dark motorcycles) where windscreen reflections had previously lifted
  them into Gray, and 2 changes on background-heavy junk crops where either
  label was meaningless.
- Added 5 new tests in `tests/unit/test_vehicle_color.py`, including one
  that pins the bug itself (`panel_row_spread_max: 0` must still
  misclassify the synthetic white van as Silver, proving the test would
  have failed before the fix).
- Cost measured at ~1ms per crop (budget in the module's own docstring is
  "low single milliseconds").
- Re-ran classification against every stored crop still on disk in your
  real database and rebuilt profiles from the result.

**Known limitation, not yet resolved:** the van's profile still shows
`Silver` overall in the live dashboard, because its colour is a *majority
vote* across all its detections, and 9 of its 11 recorded detections
belong to sessions whose crop images were already deleted during the
broken-mount incident (#1) — there is nothing left on disk to re-classify
them against. The only way to see the van correctly labelled `White` is to
clear the stored data (`alpr/bin/python scripts/clear_data.py`) and
re-run the cameras so every detection is re-classified with the fixed
code. **This step has not been done — it deletes all currently stored
events and images, and that's a decision for you to make, not something
done automatically.**

---

## Files changed, by area

**New:**
- `run` — the unified launcher (item 5)
- `src/runtime/` (`device.py`, `errors.py`, `__init__.py`) — was already
  authored in the working tree from prior work outside this session; the
  Docker image had simply gone stale relative to it (item 1)
- `tests/unit/test_device_resolver.py`, `test_ocr_backend_selection.py`,
  `test_ocr_engine_failfast.py`, `test_rapidocr_result_shapes.py` — same
  as above, pre-existing working-tree content that the stale image was
  missing
- `docs/SESSION_CHANGELOG.md` — this file

**Modified:**
- `docker-compose.yml` — live source mounts, test fixture mounts (item 5)
- `docker/start.sh` — delegates to `./run up` (item 5)
- `.env` — corrected UID/GID (item 1, auto-maintained by `./run` from
  here on)
- `pytest.ini` — cache directory fix (item 5)
- `README.md` — quick-start rewritten (item 5)
- `config/config.yaml` — added `video.realtime_playback` (item 7),
  `vehicle_color.panel_row_spread_max` (item 8)
- `src/dashboard/trajectory_app.py` — caching, fragments, frame-read
  efficiency (items 2, 3)
- `src/database/vehicle_profiles.py` — existence-aware image selection
  (item 6)
- `src/capture/frame_capture.py`, `src/cameras/sources.py` — real-time
  playback pacing (item 7)
- `src/classification/vehicle_color.py` — panel-row brightness sampling
  (item 8)
- `tests/integration/test_dashboard_smoke.py` — 5 new tests (item 3)
- `tests/integration/test_vehicle_profiles.py` — 2 new regression tests,
  3 existing tests fixed to use real image files instead of synthetic
  filenames that were never real (item 6)
- `tests/unit/test_vehicle_color.py` — 5 new tests (item 8)

---

## Tests that fail and are *not* part of this work

Three tests fail consistently, before and after every change above, in
both Docker and the native virtualenv — confirmed genuine, pre-existing
bugs unrelated to anything in this document:

- `tests/unit/test_parseq_dataset.py::test_build_dataloaders_creates_batches`
  — the test's own temp fixture produces no labeled samples.
- `tests/unit/test_super_resolution.py::TestSuperResolutionFallback::test_narrow_crop_with_missing_model_returns_original`
- `tests/unit/test_super_resolution.py::TestSuperResolutionFallback::test_property6_fallback_when_model_missing`
  — these two assert that a *missing* Real-ESRGAN model returns a crop
  unchanged, but the model is actually present and upscaling 4x, so the
  output shapes don't match what the test expects.

These were flagged, not fixed, since fixing test logic wasn't part of any
of the requests above. Say the word if you'd like them addressed too.
