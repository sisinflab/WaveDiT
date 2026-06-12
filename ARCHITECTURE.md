# WaveDiT Studio - macOS app architecture

WaveDiT Studio is a self-contained, native macOS (Apple Silicon) desktop application for
generating and exploring age-conditioned synthetic 3D brain MRI with WaveDiT. It is built
from this directory and distributed as a DMG. Research artifact only: synthetic images,
not a medical device, not for clinical use.

## Design constraints

- No Electron. The UI runs in the system WKWebView (Apple WebKit) through `pywebview`;
  the backend is plain Python running natively on arm64 with PyTorch MPS.
- Fully self-contained bundle: every Python dependency is frozen into the .app by
  PyInstaller. The only thing downloaded at runtime is model weights, from
  `danesed/WaveDiT` on Hugging Face.
- The in-browser 3D viewer (Niivue 0.69.0, UMD build) is vendored at
  `studio/ui/vendor/niivue.umd.js` (sha256
  `R7iWt37Epb4+8ZScM605O29FYpkh8ybCeQALz1G9tK8=`, base64). No CDN at runtime; the app
  works offline once weights are downloaded.
- The `wavedit` package is vendored at `studio/wavedit` (copy of `space/wavedit`, which
  already carries the lazy-wandb patch and pruned `evaluation/` + `data/` subtrees).
  One extra patch: `sample()` enables autocast on `mps` as well as `cuda`
  (`models/wavelet_flow_matching.py`). Internal imports are absolute (`from wavedit...`),
  so `studio/` must be on `sys.path` and the package imports as top-level `wavedit`.

## Component map

```
macos/
  ARCHITECTURE.md          this file
  README.md                user-facing: build, install, Gatekeeper, troubleshooting
  build.sh                 one-command build on an Apple Silicon Mac -> DMG
  requirements.txt         pinned runtime deps (torch 2.8.0 arm64, pywebview, ...)
  packaging/
    wavedit_studio.spec    PyInstaller spec (windowed .app bundle)
    make_icon.py           renders the app icon + DMG background (numpy, no Pillow)
    make_dmg.sh            staging dir -> codesign ad-hoc -> hdiutil UDZO DMG
  studio/
    __init__.py            __version__
    __main__.py            python -m studio (delegates to main.main())
    main.py                entry point: server thread + pywebview window; headless mode
    server.py              stdlib ThreadingHTTPServer: static UI + JSON API + SSE
    engine.py              device policy, model registry, generation jobs, progress, sweep
    weights.py             HF weights manager: remote list, streamed download, delete
    library.py             on-disk library of generations (nii.gz + meta.json + thumb.png)
    settings.py            persisted user settings (JSON)
    sysinfo.py             chip / RAM / OS / torch device detection
    minipng.py             minimal grayscale PNG encoder (stdlib zlib only)
    paths.py               data dirs (macOS: ~/Library/Application Support/WaveDiT Studio)
    wavedit/               vendored wavedit package (see above)
    ui/
      index.html           single-page UI
      app.css              design system + layout (dark default, light via media query)
      app.js               UI logic, SSE client, Niivue viewer integration (vanilla JS)
      vendor/niivue.umd.js vendored viewer library (exposes global `niivue`)
```

## Runtime model

`main.py` starts `server.py` on `127.0.0.1:<random free port>` in a daemon thread, then
opens a `pywebview` window at that URL. Modes and overrides (all env vars):

- `WAVEDIT_STUDIO_HEADLESS=1`  server only, no window (Linux dev / CI smoke tests).
- `WAVEDIT_STUDIO_DEVICE`      force `mps` | `cuda` | `cpu` (default: mps > cuda > cpu).
- `WAVEDIT_STUDIO_CKPT_DIR`    use an existing checkpoints dir instead of downloading.
- `WAVEDIT_STUDIO_DATA_DIR`    override the app-support data dir (tests).
- `WAVEDIT_STUDIO_PORT`        fixed port (default: ephemeral).
- `--selfcheck` CLI flag       print versions + device, exit 0 (used by build.sh).

`WAVEDIT_NA_BACKEND=torch` and `PYTORCH_ENABLE_MPS_FALLBACK=1` are set in `main.py`
before importing torch/wavedit.

Models are loaded lazily on first use (desktop app: cold start must stay fast), cached
in a registry keyed by checkpoint filename, and kept resident. `num_flow_steps_sampling`
is a mutable attribute set per request. Only one generation job runs at a time (a lock);
the API returns 409 if busy.

### Generation pipeline (engine.py)

1. Resolve checkpoint path (local checkpoints dir; error if not downloaded).
2. Build once: `Config.from_dict(ck["config"])`, `build_model(...)`,
   `load_model_weights(model, ck)`, `.to(device).eval()`.
3. Seed: `torch.manual_seed(seed)` (+ `torch.cuda.manual_seed_all` on cuda).
4. Progress: register a forward hook on `model.backbone`. Total backbone calls
   `nfe_total = velocity_calls * (2 if cfg_scale != 1.0 else 1)` where
   `velocity_calls = 2 * steps - 1` for heun, `steps` for euler. The hook increments a
   counter, records timestamps for ETA, publishes SSE progress, and raises
   `GenerationCancelled` when the job's cancel flag is set. The hook is always removed
   in a `finally`.
5. `model.sample(num_samples=1, raw_conditions={"age": age}, cfg_scale, sampler,
   morpheus_scale=None if morpheus == 1.0 else morpheus, cfg_rescale=0.7,
   autocast_dtype=<precision policy>)`.
6. Precision policy: `float32` is the default everywhere. If the user enables
   performance mode (`bf16`), pass `torch.bfloat16`; on failure, fall back to float32
   for the retry and persist `bf16_ok=false` in settings.
7. Postprocess: `clamp((v+1)/2, 0, 1)`, center-crop from the checkpoint's
   `cfg.data.image_size` to `(182, 218, 182)` via `center_crop_bounds`, float32 numpy.
8. Persist via library.py and publish `gen_done`.

Memory badge: on cuda use `torch.cuda.max_memory_allocated()`; on mps sample
`torch.mps.current_allocated_memory()` from a 200 ms monitor thread during the job and
report the max. Per-NFE wall time is recorded per (model, precision) in settings as
calibration so the UI can show live time estimates.

### Library layout (library.py)

```
<data dir>/
  settings.json
  checkpoints/                  WaveDiT-Base.pth, WaveDiT-FinePatch.pth, ...
  library/<id>/
    volume.nii.gz               float32, (182, 218, 182), intensities in [0, 1]
    thumb.png                   mid-axial slice, 8-bit grayscale, rot90 like the Space
    meta.json                   full provenance (see item shape below)
```

`id` is `YYYYMMDD-HHMMSS-<seed>-<4 hex chars>`. The float32 file is served to the
viewer directly (localhost, size is irrelevant) and copied on export.

## HTTP API (server.py, JSON unless noted)

- `GET  /`                     index.html; `GET /<static>` from `studio/ui` only.
- `GET  /api/state`            app snapshot: `{version, device: {device, chip, ram_gb,
  torch, os, os_version, python}, weights: [...], age_range: [lo, hi], settings,
  calibration, library_count, busy}`. Control defaults live client-side (the UI
  falls back to settings.default_model plus the Space defaults).
- `GET  /api/weights`          `[{file, label, size_mb, downloaded, downloading}]`,
  remote list from one bounded urllib call to the Hub tree endpoint (TTL cached,
  negatively cached on failure, offline-safe: falls back to local files).
- `POST /api/weights/download` `{file}` -> `{ok}`; streamed urllib download to
  `checkpoints/<file>.part` then atomic rename; progress over SSE.
- `POST /api/weights/delete`   `{file}` -> `{ok}` (refuses while downloading/loaded).
- `POST /api/generate`         `{model, age, seed, steps, cfg_scale, sampler, morpheus}`
  -> `{job_id}` or 409 `{error}`.
- `POST /api/sweep`            `{model, age_start, age_end, frames, seed, steps, ...}`
  -> `{job_id}`; one generation per frame, shared seed.
- `POST /api/cancel`           `{job_id}` -> `{ok}`.
- `GET  /api/library`          newest-first `[item]`.
- `POST /api/library/delete`   `{id}` -> `{ok}`.
- `POST /api/export`           `{id}` -> native save dialog via pywebview on macOS;
  headless fallback: `{path}` of a copy in `<data dir>/exports/`.
- `POST /api/settings`         merge + persist, returns full settings.
- `GET  /api/events`           SSE stream (see below).
- `GET  /volumes/<id>.nii.gz`  the float32 volume (Content-Type application/gzip).
- `GET  /thumbs/<id>.png`      the thumbnail.

Item shape (`meta.json`, `gen_done`, `/api/library`):

```json
{"id": "...", "created": 1765500000.0, "model": "WaveDiT-Base.pth",
 "model_label": "Base", "age": 50, "seed": 42, "steps": 10, "cfg_scale": 1.0,
 "sampler": "heun", "morpheus": 1.0, "precision": "float32", "wall_s": 12.3,
 "nfe_total": 19, "peak_mem_gb": 6.1, "device": "mps", "sweep_id": null,
 "vol_url": "/volumes/<id>.nii.gz", "thumb_url": "/thumbs/<id>.png"}
```

### SSE events (`event:` name + `data:` JSON)

- `gen_start`     `{job_id, nfe_total, params}`
- `gen_progress`  `{job_id, nfe_done, nfe_total, pct, eta_s, elapsed_s}`
- `gen_done`      `{job_id, item}`
- `gen_error`     `{job_id, message}` (also emitted on cancel, `cancelled: true`)
- `sweep_start`   `{job_id, frames, params}`
- `sweep_progress` `{job_id, frame, frames, pct, eta_s}`
- `sweep_frame_done` `{job_id, frame, item}`
- `sweep_done`    `{job_id, sweep_id, ids}`
- `weights_progress` `{file, mb_done, mb_total, pct, speed_mbps}`
- `weights_done`  `{file}`
- `weights_error` `{file, message}`

The server keeps a broadcast hub; each SSE client gets a queue. Events are also
buffered (last gen/sweep state) so a UI that reconnects can resync via `/api/state`.

## Security posture

The server binds to 127.0.0.1 only. Static file serving is restricted to the `ui/`
tree (no `..` traversal; resolved paths must stay inside). `/volumes` and `/thumbs`
validate the id against `[A-Za-z0-9_-]+`. No auth: localhost, single user, no secrets.

## Build pipeline (build.sh, runs on the Mac)

1. Guards: `uname -m` is arm64, macOS >= 13, network present.
2. Python: use `uv` if present, else offer to install it (asks first), else fall back
   to a system `python3` >= 3.11. Creates `.venv-build` with Python 3.12.
3. `pip install -r requirements.txt` plus `pyinstaller` (build-only dep).
4. Verify the vendored `niivue.umd.js` sha256; re-fetch from jsdelivr if missing.
5. `python packaging/make_icon.py` -> iconset PNGs + DMG background; `iconutil` -> .icns.
6. `pyinstaller packaging/wavedit_studio.spec` -> `dist/WaveDiT Studio.app`.
7. Smoke test: run the bundled binary with `--selfcheck`.
8. `codesign --force --deep -s -` (ad-hoc) on the .app.
9. `packaging/make_dmg.sh`: staging dir with the .app + `/Applications` symlink +
   background, `hdiutil create -format UDZO` -> `dist/WaveDiT-Studio-<version>.dmg`.
10. Print install notes (right-click -> Open on first launch, or
    `xattr -dr com.apple.quarantine`).

Only tools that are part of stock macOS are required (codesign, iconutil, hdiutil,
sips, osascript). No Xcode, no Homebrew.

## UI principles

Single page, vanilla JS, no framework. System font stack (SF Pro on macOS). Dark
theme default, light theme via `prefers-color-scheme`. Indigo (#6366f1) to violet
(#a855f7) accent gradient, consistent with the WaveDiT brand. Layout: left control
sidebar (age hero control, model picker, generate, advanced disclosure, presets,
aging time-lapse), main viewer (Niivue multiplanar + 3D render with clip plane,
colormap/gamma controls, generation overlay with real per-NFE progress and ETA),
bottom library shelf (thumbnails, reuse settings, export, delete). First-run
onboarding modal handles weights download with live progress. A persistent footer
line states the research-only scope.
