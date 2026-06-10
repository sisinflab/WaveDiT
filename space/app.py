"""WaveDiT - 3D Brain MRI Generator (Hugging Face ZeroGPU Gradio Space).

Single-page Gradio app:
  * pick an age (and optionally model / seed / steps / CFG / sampler / Morpheus),
  * generate a full-resolution synthetic 3D brain MRI on a ZeroGPU GPU,
  * explore it in-browser with an MRIcroGL-like Niivue viewer (triplane + 3D render),
  * sweep an "aging time-lapse", apply presets, switch colormap/gamma live,
  * download the float32 NIfTI and read an always-on generation/VRAM badge.

The wavedit/ package is VENDORED into this Space (do not pip-install it: its
pyproject pins torch==2.6.0, which is unsupported on ZeroGPU >= 2.8). The only
source patch is a lazy `import wandb` in BOTH
  - wavedit/models/wavelet_flow_matching.py  (used at sampling-import time), and
  - wavedit/training/trainer.py               (pulled in transitively because
    wavedit/training/__init__.py imports Trainer, and app.py imports
    load_model_weights from wavedit.training.checkpoint).

Research artifact only - synthetic images, not a medical device, not for clinical use.
"""

from __future__ import annotations

import os

# CRITICAL: select the pure-PyTorch neighborhood-attention fallback BEFORE importing
# wavedit (NATTEN is unavailable on Spaces; the torch backend is numerically equivalent).
os.environ.setdefault("WAVEDIT_NA_BACKEND", "torch")

import base64
import gzip
import html as html_lib
import random
import time
import traceback
from pathlib import Path

import gradio as gr
import nibabel as nib
import numpy as np
import torch

try:
    import spaces  # True no-op off ZeroGPU; gates the real GPU on @spaces.GPU.
    _HAS_SPACES = True
except Exception:  # pragma: no cover - running fully off-platform
    _HAS_SPACES = False

    class _SpacesShim:
        @staticmethod
        def GPU(*dargs, **dkwargs):
            def _wrap(fn):
                return fn

            # Support both @spaces.GPU and @spaces.GPU(duration=...)
            if dargs and callable(dargs[0]) and not dkwargs:
                return dargs[0]
            return _wrap

    spaces = _SpacesShim()  # type: ignore

from huggingface_hub import hf_hub_download

from wavedit import Config
from wavedit.models import build_model

# IMPORT load_model_weights DIRECTLY from .training.checkpoint. NOTE: this still runs
# wavedit/training/__init__.py (which imports trainer.py); the vendored trainer.py has
# the lazy-wandb patch, so this import succeeds even when wandb is not installed.
from wavedit.training.checkpoint import load_model_weights
from wavedit.generation.generator import center_crop_bounds

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
HF_REPO = "danesed/WaveDiT"
HF_REVISION = "main"

CHECKPOINTS = {
    "Base (fast)": "WaveDiT-Base.pth",
    "FinePatch (detailed)": "WaveDiT-FinePatch.pth",
}
DEFAULT_MODEL = "Base (fast)"

# Standard MNI-like target grid; FULL_SIZE is read per-checkpoint at build time so a
# future Deep/Wide variant with a different image_size still crops correctly.
DEFAULT_FULL_SIZE = (224, 224, 224)
CROP_SIZE = (182, 218, 182)

# Fallbacks if a checkpoint somehow lacks an age range (the real ones carry 6..95).
AGE_MIN_FALLBACK, AGE_MAX_FALLBACK = 6, 95
AGE_DEFAULT = 72
SEED_DEFAULT = 42
SEED_MAX = 2_147_483_647

STEPS_DEFAULT = 10
STEPS_MIN, STEPS_MAX = 1, 200

# Aging time-lapse runs many frames; keep each frame within one short GPU window and
# bound total session work. Per-frame steps are clamped to SWEEP_STEPS_MAX and the
# frame count is bounded so a worst-case FinePatch sweep never overruns the per-call
# @spaces.GPU budget (each frame is its OWN GPU call -- see gpu_sweep_frame).
SWEEP_FRAMES_MIN, SWEEP_FRAMES_MAX = 3, 8
SWEEP_STEPS_MAX = 50

CFG_MIN, CFG_MAX, CFG_DEFAULT = 1.0, 8.0, 1.0
CFG_RESCALE = 0.7  # fixed; only active when cfg_scale != 1.0
MORPHEUS_DEFAULT = 1.0

SAMPLERS = ["Heun", "Euler"]
SAMPLER_MAP = {"Heun": "heun", "Euler": "euler"}

COLORMAPS = ["gray", "bone", "viridis", "plasma", "inferno", "magma", "hot", "cubehelix"]
DELIGHT_COLORMAPS = ["viridis", "plasma", "bone", "magma"]

SPACE_VERSION = "0.1.0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HERE = Path(__file__).resolve().parent
VIEWER_TEMPLATE_PATH = HERE / "viewer_template.html"
HERO_NII_PATH = HERE / "assets" / "wavedit_hero_finepatch_age72_seed42.nii.gz"
DOWNLOAD_DIR = Path(os.environ.get("WAVEDIT_OUT_DIR", "/tmp/wavedit_outputs"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Links / copy
# --------------------------------------------------------------------------- #
LINK_PAPER = "https://arxiv.org/abs/2606.08670"
LINK_CODE = "https://github.com/sisinflab/WaveDiT"
LINK_PROJECT = "https://danesed.github.io/wavedit-page/"
LINK_MODEL = "https://huggingface.co/danesed/WaveDiT"

CITATION = """@misc{danese2026waveditdistributionawarewaveletflow,
  title         = {WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis},
  author        = {Danilo Danese and Angela Lombardi and Giuseppe Fasano and Matteo Attimonelli and Tommaso Di Noia},
  year          = {2026},
  eprint        = {2606.08670},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.08670}
}"""

# --------------------------------------------------------------------------- #
# Viewer template
# --------------------------------------------------------------------------- #
_INLINE_VIEWER_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html,body{margin:0;height:100%;background:#0a0e14;overflow:hidden}
#gl{width:100%;height:100%;display:block}
#err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
color:#ffd0d0;background:#0a0e14;font-family:system-ui;padding:24px;text-align:center}</style></head>
<body><canvas id="gl"></canvas>
<div id="err">This viewer needs WebGL2. Use a recent browser, or the Download .nii.gz button.<br>
<code id="errmsg"></code></div>
<script type="module">
const showErr=(m)=>{const e=document.getElementById("err");document.getElementById("errmsg").textContent=m||"";e.style.display="flex"};
let webgl2ok=true;let nv=null;
try{const t=document.createElement("canvas").getContext("webgl2");if(!t){webgl2ok=false;showErr("getContext(webgl2) returned null.")}}catch(e){webgl2ok=false;showErr(String(e))}
if(webgl2ok){try{const mod=await import("https://esm.sh/@niivue/niivue@0.69.0");
const {Niivue,SLICE_TYPE,SHOW_RENDER,DRAG_MODE}=mod;
nv=new Niivue({backColor:[0.04,0.05,0.07,1],dragMode:DRAG_MODE.slicer3D,
multiplanarShowRender:SHOW_RENDER.ALWAYS,isResizeCanvas:true});
await nv.attachTo("gl");
await nv.loadVolumes([{url:"__DATA_URL__",name:"brain.nii.gz",colormap:"__COLORMAP__"}]);
nv.setSliceType(SLICE_TYPE.MULTIPLANAR);
if(nv.volumes.length){nv.setColormap(nv.volumes[0].id,"__COLORMAP__")}
nv.setClipPlane([0.3,180,20]);nv.setInterpolation(false);nv.drawScene();
}catch(e){showErr("Could not load the 3D viewer library (network/CDN). Use the Download .nii.gz button. "+String(e&&e.message?e.message:e))}}
window.addEventListener("message",(ev)=>{const d=ev&&ev.data;if(!d||!nv||!nv.volumes||!nv.volumes.length)return;try{
if(d.type==="colormap"&&typeof d.value==="string")nv.setColormap(nv.volumes[0].id,d.value);
else if(d.type==="gamma")nv.setGamma(parseFloat(d.value));
else if(d.type==="clip"&&Array.isArray(d.value))nv.setClipPlane(d.value);
else if(d.type==="reset"){nv.setClipPlane([0.3,180,20])}
nv.drawScene()}catch(e){}});
</script></body></html>"""


def _load_viewer_template() -> str:
    if VIEWER_TEMPLATE_PATH.exists():
        try:
            return VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    return _INLINE_VIEWER_TEMPLATE


VIEWER_TEMPLATE = _load_viewer_template()


def _render_iframe(data_url: str, colormap: str) -> str:
    """Build the Niivue <iframe srcdoc> HTML string for a given data URL + colormap."""
    doc = VIEWER_TEMPLATE.replace("__DATA_URL__", data_url).replace("__COLORMAP__", colormap)
    # srcdoc must be attribute-safe: HTML-escape (quotes included). The template avoids
    # apostrophes, but escaping makes the embed robust regardless.
    srcdoc = html_lib.escape(doc, quote=True)
    return (
        f'<iframe title="WaveDiT 3D viewer" sandbox="allow-scripts" '
        f'style="width:100%;height:620px;border:0;border-radius:4px;background:#0a0e14" '
        f'srcdoc="{srcdoc}"></iframe>'
    )


def _placeholder_iframe(message: str) -> str:
    msg = html_lib.escape(message, quote=True)
    doc = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<style>html,body{margin:0;height:100%;display:flex;align-items:center;'
        'justify-content:center;background:#0a0e14;color:#7fd1ff;'
        'font-family:system-ui;text-align:center;padding:24px}</style></head>'
        f"<body><div>{msg}</div></body></html>"
    )
    srcdoc = html_lib.escape(doc, quote=True)
    return (
        f'<iframe title="WaveDiT 3D viewer" sandbox="allow-scripts" '
        f'style="width:100%;height:620px;border:0;border-radius:4px;background:#0a0e14" '
        f'srcdoc="{srcdoc}"></iframe>'
    )


# --------------------------------------------------------------------------- #
# Model registry -- EAGER module-scope preload (ZeroGPU contract).
#
# ZeroGPU runs every @spaces.GPU call in a forked SUBPROCESS, and mutations to
# module-scope state inside that call do NOT propagate back to the parent. So a
# cache populated *inside* the GPU function would be thrown away on every call and
# rebuilt from scratch -- burning the user's (very limited) GPU quota on one-time
# work and hitting the documented "less efficient" CUDA-transfer-in-fn anti-pattern.
#
# The fix the docs mandate: build BOTH checkpoints and move them to cuda at the
# MODULE LEVEL (parent process, where `import spaces` enables CUDA emulation so
# `.to("cuda")` succeeds and the placement is inherited by every forked child).
# Inside the decorated fn we then ONLY set num_flow_steps and call sample().
# num_flow_steps_sampling is a mutable instance attribute (verified line 69 of
# wavelet_flow_matching.py, consumed at line 275), so the steps slider is a cheap
# in-place overwrite -- no rebuild, ever.
#
# Resident weights are tiny relative to the 48 GB "large" GPU: Base ~3 GB + FinePatch
# ~8.4 GB peak fit comfortably; do NOT request xlarge.
# --------------------------------------------------------------------------- #
_MODEL_REGISTRY: dict[str, torch.nn.Module] = {}
_CKPT_META: dict[str, dict] = {}


def _checkpoint_path(model_label: str) -> str:
    filename = CHECKPOINTS[model_label]
    # Local-dev escape hatch: WAVEDIT_LOCAL_CKPT_DIR lets you run the Space against
    # on-disk checkpoints (no Hub round-trip). Unset on the deployed Space -> uses the Hub.
    local_dir = os.environ.get("WAVEDIT_LOCAL_CKPT_DIR")
    if local_dir:
        local_path = Path(local_dir) / filename
        if local_path.exists():
            return str(local_path)
        print(f"[startup] WAVEDIT_LOCAL_CKPT_DIR set but {local_path} missing; falling back to the Hub.")
    return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)


def _load_checkpoint_dict(model_label: str) -> dict:
    path = _checkpoint_path(model_label)
    return torch.load(path, map_location="cpu", weights_only=True)


def _build_and_register(model_label: str) -> torch.nn.Module:
    """Download + build + load weights + move to DEVICE; register model and metadata.

    Called at MODULE SCOPE (startup), so the cuda placement is inherited by the
    forked @spaces.GPU subprocess and is never redone per-request.
    """
    ck = _load_checkpoint_dict(model_label)
    cfg = Config.from_dict(ck["config"])
    model = build_model(
        cfg,
        ck["condition_config"],
        ck["condition_ranges"],
        ck["categorical_maps"],
        ck["null_conditions"],
        num_flow_steps=STEPS_DEFAULT,
    )
    load_model_weights(model, ck)
    model.to(DEVICE).eval()

    # Read the actual output grid from the checkpoint so the center-crop is size-agnostic.
    try:
        full_size = tuple(int(s) for s in cfg.data.image_size)
    except Exception:
        full_size = DEFAULT_FULL_SIZE

    _MODEL_REGISTRY[model_label] = model
    _CKPT_META[model_label] = {
        "condition_config": ck["condition_config"],
        "condition_ranges": ck["condition_ranges"],
        "condition_ranges_age": ck.get("condition_ranges", {}).get("age", {}),
        "full_size": full_size,
    }
    return model


def get_model(model_label: str, num_flow_steps: int) -> torch.nn.Module:
    """Return the preloaded (cuda-resident) model for ``model_label``; set steps in place.

    Falls back to building on demand only if startup preload was skipped/failed (e.g.
    a fully off-platform CPU dev run that lazy-loads to keep cold start cheap).
    """
    num_flow_steps = int(max(STEPS_MIN, min(STEPS_MAX, num_flow_steps)))
    model = _MODEL_REGISTRY.get(model_label)
    if model is None:  # pragma: no cover - only when preload was skipped
        model = _build_and_register(model_label)
    model.num_flow_steps_sampling = num_flow_steps
    return model


def _full_size_for(model_label: str) -> tuple[int, int, int]:
    meta = _CKPT_META.get(model_label)
    if meta and "full_size" in meta:
        return meta["full_size"]
    return DEFAULT_FULL_SIZE


def _age_range_from_meta() -> tuple[int, int]:
    """Derive the age slider bounds from the already-loaded Base checkpoint metadata.

    Reuses the startup download (no second hf_hub_download); falls back to 6..95.
    """
    try:
        rng = _CKPT_META.get(DEFAULT_MODEL, {}).get("condition_ranges_age", {})
        lo = int(round(float(rng.get("min", AGE_MIN_FALLBACK))))
        hi = int(round(float(rng.get("max", AGE_MAX_FALLBACK))))
        if hi <= lo:
            return AGE_MIN_FALLBACK, AGE_MAX_FALLBACK
        return lo, hi
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] Could not read age bounds ({exc}); using defaults.")
        return AGE_MIN_FALLBACK, AGE_MAX_FALLBACK


def _preload_all_models() -> None:
    """Eagerly build BOTH checkpoints onto DEVICE at import time (ZeroGPU contract).

    Non-fatal: if a checkpoint cannot be fetched at startup (e.g. transient hub
    failure), get_model() will lazy-build it on first use so the UI still renders.
    """
    for label in CHECKPOINTS:
        try:
            _build_and_register(label)
            print(f"[startup] Preloaded {label} onto {DEVICE}.")
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] Could not preload {label} ({exc}); will lazy-load on demand.")


_preload_all_models()
AGE_MIN, AGE_MAX = _age_range_from_meta()


# --------------------------------------------------------------------------- #
# Post-processing helpers (CPU-side)
# --------------------------------------------------------------------------- #
def _postprocess(vol: torch.Tensor, model_label: str) -> np.ndarray:
    """(1,1,D,H,W) in [-1,1] -> (182,218,182) float32 in [0,1] (CPU numpy)."""
    vol = torch.clamp((vol.float() + 1.0) / 2.0, 0.0, 1.0)
    (d0, d1), (h0, h1), (w0, w1) = center_crop_bounds(_full_size_for(model_label), CROP_SIZE)
    vol = vol[:, :, d0:d1, h0:h1, w0:w1]
    return vol[0, 0].cpu().numpy().astype(np.float32)


def _nii_gz_bytes(arr_float01: np.ndarray, as_uint8: bool) -> bytes:
    """Serialize a (182,218,182) [0,1] array to gzipped NIfTI bytes.

    as_uint8=True  -> small payload for the in-browser viewer.
    as_uint8=False -> float32 for the scientific download.
    """
    if as_uint8:
        data = np.clip(arr_float01 * 255.0, 0, 255).round().astype(np.uint8)
    else:
        data = arr_float01.astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    raw = img.to_bytes()  # uncompressed .nii bytes
    return gzip.compress(raw, compresslevel=6)


def _data_url_from_array(arr_float01: np.ndarray) -> str:
    gz = _nii_gz_bytes(arr_float01, as_uint8=True)
    b64 = base64.b64encode(gz).decode("ascii")
    return "data:application/gzip;base64," + b64


def _write_download(arr_float01: np.ndarray, fname: str) -> str:
    gz = _nii_gz_bytes(arr_float01, as_uint8=False)  # full-precision for science
    out = DOWNLOAD_DIR / fname
    out.write_bytes(gz)
    return str(out)


def _coerce_seed(seed) -> int:
    try:
        return abs(int(seed)) % (SEED_MAX + 1)
    except (TypeError, ValueError):
        return SEED_DEFAULT


def _mid_axial_thumb(arr_float01: np.ndarray) -> np.ndarray:
    """Mid-axial slice as an HxW uint8 array for the gallery (display only)."""
    sl = arr_float01[arr_float01.shape[0] // 2, :, :]
    sl = np.rot90(sl)  # radiological-ish orientation for the thumbnail
    return (np.clip(sl, 0, 1) * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Core sampling (GPU)
# --------------------------------------------------------------------------- #
def _sample_to_array(
    model_label: str,
    age: float,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler_label: str,
    morpheus: float,
) -> tuple[np.ndarray, float, float]:
    """Run one model.sample() and return (array[0,1], wall_seconds, peak_vram_gb)."""
    steps = int(max(STEPS_MIN, min(STEPS_MAX, steps)))
    sampler = SAMPLER_MAP.get(sampler_label, "heun")
    morpheus_scale = None if abs(float(morpheus) - 1.0) < 1e-6 else float(morpheus)

    # Model is already cuda-resident from the module-scope preload (ZeroGPU contract);
    # do NOT move it here -- per the docs, CUDA transfers belong at startup, not in
    # the decorated fn. get_model() only sets the step count in place.
    model = get_model(model_label, steps)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    seed_i = _coerce_seed(seed)
    torch.manual_seed(seed_i)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed_i)

    t0 = time.perf_counter()
    with torch.no_grad():
        vol = model.sample(
            num_samples=1,
            raw_conditions={"age": float(age)},
            cfg_scale=float(cfg_scale),
            sampler=sampler,
            morpheus_scale=morpheus_scale,
            cfg_rescale=CFG_RESCALE,
            autocast_dtype=torch.bfloat16,
        )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_gb = 0.0
    if DEVICE.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    arr = _postprocess(vol, model_label)
    return arr, wall, peak_gb


# --------------------------------------------------------------------------- #
# Duration estimators for @spaces.GPU
# --------------------------------------------------------------------------- #
def _gen_duration(model_label, age, seed, steps, cfg_scale, sampler_label, morpheus, *args):
    nfe = 2 if SAMPLER_MAP.get(sampler_label, "heun") == "heun" else 1
    per_step = 0.05 if str(model_label).startswith("Base") else 0.11
    # Single generate: weights are preloaded (no build/transfer here), so this is a
    # pure sampling estimate plus a small fixed overhead. 200-step FinePatch Heun
    # (~200*2*0.11 + 15 ~= 59 s) stays well under the cap.
    return int(min(180, 15 + int(steps) * nfe * per_step))


def _sweep_frame_duration(model_label, age, frame_seed, steps,
                          cfg_scale, sampler_label, morpheus, *args):
    """Per-FRAME duration. The sweep runs each frame as its OWN @spaces.GPU call, so
    the budget only has to cover ONE frame (steps already clamped to SWEEP_STEPS_MAX).
    """
    nfe = 2 if SAMPLER_MAP.get(sampler_label, "heun") == "heun" else 1
    per_step = 0.05 if str(model_label).startswith("Base") else 0.11
    steps = min(int(steps), SWEEP_STEPS_MAX)
    return int(min(120, 15 + steps * nfe * per_step))


# --------------------------------------------------------------------------- #
# Badges / captions
# --------------------------------------------------------------------------- #
def _badge(model_label, steps, sampler_label, morpheus, wall, peak_gb):
    sampler = SAMPLER_MAP.get(sampler_label, "heun")
    short = "Base" if model_label.startswith("Base") else "FinePatch"
    abl = "  -  ABLATION (Morpheus off)" if abs(float(morpheus)) < 1e-6 else ""
    hw = "RTX Pro 6000 (ZeroGPU large)" if DEVICE.type == "cuda" else "CPU"
    vram = f"{peak_gb:.1f} GB" if peak_gb > 0 else "n/a (CPU)"
    return (
        f"{short} - {int(steps)} steps - {sampler} - 224 cubed - bf16{abl}\n"
        f"{wall:.1f} s  -  peak VRAM {vram}  -  {hw}"
    )


def _caption(model_label, age, seed, steps, sampler_label, cfg_scale, morpheus):
    sampler = SAMPLER_MAP.get(sampler_label, "heun")
    short = "Base" if model_label.startswith("Base") else "FinePatch"
    abl = " - Morpheus off" if abs(float(morpheus)) < 1e-6 else ""
    return (
        f"Synthetic - age {int(age)} - seed {_coerce_seed(seed)} - {short} "
        f"- {int(steps)} steps - {sampler} - CFG {float(cfg_scale):.1f}{abl}"
    )


def _reproduce_snippet(model_label, age, seed, steps, cfg_scale, sampler_label, morpheus):
    sampler = SAMPLER_MAP.get(sampler_label, "heun")
    filename = CHECKPOINTS[model_label]
    morpheus_arg = "None" if abs(float(morpheus) - 1.0) < 1e-6 else f"{float(morpheus):.2f}"
    return f'''import os
os.environ["WAVEDIT_NA_BACKEND"] = "torch"  # set before importing wavedit

import torch
from huggingface_hub import hf_hub_download
from wavedit import Config
from wavedit.models import build_model
from wavedit.training.checkpoint import load_model_weights
from wavedit.generation.generator import center_crop_bounds

path = hf_hub_download("{HF_REPO}", "{filename}", revision="{HF_REVISION}")
ck = torch.load(path, map_location="cpu", weights_only=True)

cfg = Config.from_dict(ck["config"])
model = build_model(cfg, ck["condition_config"], ck["condition_ranges"],
                    ck["categorical_maps"], ck["null_conditions"], num_flow_steps={int(steps)})
load_model_weights(model, ck)
model.to("cuda").eval()

torch.manual_seed({_coerce_seed(seed)})
vol = model.sample(num_samples=1, raw_conditions={{"age": {float(age):.1f}}},
                   cfg_scale={float(cfg_scale):.1f}, sampler="{sampler}",
                   morpheus_scale={morpheus_arg}, cfg_rescale={CFG_RESCALE},
                   autocast_dtype=torch.bfloat16)

vol = torch.clamp((vol.float() + 1.0) / 2.0, 0.0, 1.0)
(d0, d1), (h0, h1), (w0, w1) = center_crop_bounds((224, 224, 224), (182, 218, 182))
arr = vol[:, :, d0:d1, h0:h1, w0:w1][0, 0].cpu().numpy()  # (182, 218, 182) in [0, 1]'''


def _filename(model_label, age, seed, steps):
    short = "base" if model_label.startswith("Base") else "finepatch"
    return f"wavedit_{short}_age{int(age)}_seed{_coerce_seed(seed)}_steps{int(steps)}.nii.gz"


# --------------------------------------------------------------------------- #
# Living microcopy
# --------------------------------------------------------------------------- #
def age_microcopy(age) -> str:
    a = int(age)
    if a <= 12:
        body = "developing brain - ventricles typically tight, cortex still maturing."
    elif a <= 25:
        body = "young adult - typically near peak cortical volume."
    elif a <= 59:
        body = "adult brain - typically stable morphology."
    elif a <= 79:
        body = "older adult - ventricles typically begin to widen."
    else:
        body = "elderly - typically more pronounced atrophy and wider ventricles."
    return f"**Age {a}:** {body}"


def cfg_microcopy(cfg_scale) -> str:
    if abs(float(cfg_scale) - 1.0) < 1e-6:
        return "Guidance off (CFG = 1.0). Std-rescaling activates only above 1.0."
    return "Classifier-free guidance on; std-rescale (cfg_rescale = 0.7) tames magnitude growth."


def steps_microcopy(steps, model_label=DEFAULT_MODEL) -> str:
    s = int(steps)
    if s > 50 and not str(model_label).startswith("Base"):
        return "High step count - a FinePatch run can take up to ~90 s."
    if s > 100:
        return "Very high step count - this may take a while."
    return "More steps = smoother integration. Default 10 is fast and cheap."


# --------------------------------------------------------------------------- #
# Gradio event handlers
# --------------------------------------------------------------------------- #
@spaces.GPU(duration=_gen_duration)
def gpu_generate(model_label, age, seed, steps, cfg_scale, sampler_label, morpheus):
    """GPU work only: returns (array, wall, peak_gb). CPU-side packaging happens outside."""
    return _sample_to_array(model_label, age, seed, steps, cfg_scale, sampler_label, morpheus)


def generate(model_label, age, seed, steps, cfg_scale, sampler_label, morpheus, colormap):
    """Top-level generate handler: runs the GPU fn, packages on CPU, builds the viewer."""
    try:
        arr, wall, peak_gb = gpu_generate(
            model_label, age, seed, steps, cfg_scale, sampler_label, morpheus
        )
    except Exception as exc:  # OOM / quota / hub / anything: stay non-fatal.
        # NOTE: do NOT call torch.cuda.empty_cache() here -- this runs in the MAIN
        # process where no real GPU is attached (the @spaces.GPU subprocess has
        # already torn down), so it frees nothing and risks a blocked cuda lazy-init.
        msg = str(exc)
        low = msg.lower()
        friendly = "Generation failed. "
        if "out of memory" in low or "oom" in low:
            friendly += "Out of GPU memory - try fewer steps or the Base model."
        elif "quota" in low or "gpu task" in low:
            friendly += "ZeroGPU quota reached - wait a moment and try again."
        else:
            friendly += "Please try again."
        traceback.print_exc()
        # Keep the previous viewer (gr.update no-op) and surface a friendly status.
        return (
            gr.update(),                              # viewer unchanged
            gr.update(value=f"WARNING: {friendly}"),  # status
            gr.update(),                              # badge
            gr.update(),                              # caption
            gr.update(),                              # download
        )

    seed_i = _coerce_seed(seed)
    data_url = _data_url_from_array(arr)
    iframe = _render_iframe(data_url, colormap)
    badge = _badge(model_label, steps, sampler_label, morpheus, wall, peak_gb)
    caption = _caption(model_label, age, seed_i, steps, sampler_label, cfg_scale, morpheus)
    dl_path = _write_download(arr, _filename(model_label, age, seed_i, steps))
    status = f"Done in {wall:.1f} s. Drag the 3D pane to cut into the brain."

    return (
        iframe,
        status,
        badge,
        caption,
        gr.update(value=dl_path, visible=True),
    )


@spaces.GPU(duration=_sweep_frame_duration)
def gpu_sweep_frame(model_label, age, frame_seed, steps,
                    cfg_scale, sampler_label, morpheus):
    """ONE aging-sweep frame on the GPU. Each frame is its own short @spaces.GPU call.

    Running per-frame (rather than one long call over all frames) keeps every GPU
    window comfortably inside its declared duration -- a 12x200-step FinePatch sweep
    in a single call would exceed any honest budget and ZeroGPU would kill it midway.
    Returns just the CPU array; main-process run_sweep packages + drives gr.Progress.
    """
    arr, _, _ = _sample_to_array(
        model_label, age, frame_seed, steps, cfg_scale, sampler_label, morpheus
    )
    return arr


def run_sweep(model_label, start_age, end_age, frames, fix_seed, seed,
              steps, cfg_scale, sampler_label, morpheus, colormap,
              progress=gr.Progress()):
    """Drive the aging time-lapse from the MAIN process.

    gr.Progress lives in the main Gradio process; calling it from inside an
    @spaces.GPU subprocess is not a documented-supported path (args/returns are
    pickled across the fork). So progress is driven HERE and each frame's GPU work
    is a separate gpu_sweep_frame() call. Per-frame steps are clamped to
    SWEEP_STEPS_MAX and frames to SWEEP_FRAMES_MAX so total session work stays bounded.
    """
    frames = int(max(SWEEP_FRAMES_MIN, min(SWEEP_FRAMES_MAX, frames)))
    sweep_steps = int(min(int(steps), SWEEP_STEPS_MAX))
    ages = np.linspace(float(start_age), float(end_age), frames)
    base_seed = _coerce_seed(seed)

    results = []
    try:
        for i, a in enumerate(ages):
            progress(i / frames, desc=f"Aging {i + 1}/{frames} - age {int(round(a))}")
            frame_seed = base_seed if fix_seed else (base_seed + i) % (SEED_MAX + 1)
            arr = gpu_sweep_frame(
                model_label, float(a), frame_seed, sweep_steps,
                cfg_scale, sampler_label, morpheus,
            )
            results.append((int(round(a)), arr))
        progress(1.0, desc="Done")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if results:
            # Salvage whatever frames completed before the failure.
            pass
        else:
            return (
                gr.update(),                                            # viewer
                gr.update(),                                            # gallery
                gr.update(value=f"WARNING: Time-lapse failed: {exc}"),  # status
                gr.update(),                                            # frame slider
                {},                                                     # sweep state
            )

    # Gallery of mid-axial slices labelled by age (instant figure material).
    gallery = [(_mid_axial_thumb(arr), f"age {a}") for a, arr in results]
    # Show the first frame in the main viewer; precompute data URLs for the scrubber.
    urls = [_data_url_from_array(arr) for _, arr in results]
    labels = [a for a, _ in results]
    iframe = _render_iframe(urls[0], colormap)

    note = ("Same random seed across ages - only the age condition changes."
            if fix_seed else "Seed varies per frame.")
    clamp_note = (f" (sweep uses {sweep_steps} steps/frame)"
                  if sweep_steps < int(steps) else "")
    status = (f"Time-lapse ready: {len(results)} frames. {note}{clamp_note} "
              "Qualitative, not a validated atrophy measurement.")

    return (
        iframe,
        gr.update(value=gallery, visible=True),
        gr.update(value=status),
        gr.update(visible=True, maximum=len(urls) - 1, value=0),
        {"urls": urls, "labels": labels, "colormap": colormap},
    )


def show_sweep_frame(idx, sweep_state):
    if not sweep_state or "urls" not in sweep_state:
        return gr.update()
    urls = sweep_state["urls"]
    cmap = sweep_state.get("colormap", "gray")
    i = int(max(0, min(len(urls) - 1, idx)))
    return _render_iframe(urls[i], cmap)


# Random buttons
def random_age():
    # 20% weighted to the dramatic extremes for more striking brains.
    if random.random() < 0.2:
        a = random.choice(
            list(range(AGE_MIN, min(AGE_MIN + 13, AGE_MAX)))
            + list(range(max(AGE_MIN, AGE_MAX - 15), AGE_MAX + 1))
        )
    else:
        a = random.randint(AGE_MIN, AGE_MAX)
    return a


def random_seed():
    return random.randint(0, SEED_MAX)


# Presets: return updates for (model, age, seed, steps, cfg, sampler, morpheus, colormap)
def preset_child():
    return DEFAULT_MODEL, 8, SEED_DEFAULT, STEPS_DEFAULT, CFG_DEFAULT, "Heun", MORPHEUS_DEFAULT, "gray"


def preset_prime():
    return DEFAULT_MODEL, 35, SEED_DEFAULT, STEPS_DEFAULT, CFG_DEFAULT, "Heun", MORPHEUS_DEFAULT, "gray"


def preset_elder():
    return DEFAULT_MODEL, 82, SEED_DEFAULT, STEPS_DEFAULT, CFG_DEFAULT, "Heun", MORPHEUS_DEFAULT, "bone"


def preset_showcase():
    return "FinePatch (detailed)", 80, SEED_DEFAULT, 80, CFG_DEFAULT, "Heun", MORPHEUS_DEFAULT, "plasma"


def preset_ablation():
    return DEFAULT_MODEL, AGE_DEFAULT, SEED_DEFAULT, STEPS_DEFAULT, CFG_DEFAULT, "Heun", 0.0, "gray"


def preset_surprise():
    a = random_age()
    s = random_seed()
    cmap = random.choice(DELIGHT_COLORMAPS)
    model = random.choice(list(CHECKPOINTS.keys()))
    return model, a, s, STEPS_DEFAULT, CFG_DEFAULT, "Heun", MORPHEUS_DEFAULT, cmap


# Initial viewer (hero brain if present; otherwise a friendly placeholder).
def _initial_viewer_html() -> str:
    if HERO_NII_PATH.exists():
        try:
            gz = HERO_NII_PATH.read_bytes()
            b64 = base64.b64encode(gz).decode("ascii")
            data_url = "data:application/gzip;base64," + b64
            return _render_iframe(data_url, "gray")
        except Exception:
            pass
    return _placeholder_iframe(
        "Pick an age and press Generate brain to synthesize a 3D MRI you can rotate and slice."
    )


# --------------------------------------------------------------------------- #
# Theme + CSS
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="linear-gradient(180deg,#0d1117 0%,#10131c 100%)",
    block_background_fill="#161b26",
    block_radius="3px",
    button_primary_background_fill="linear-gradient(90deg,#6366f1,#a855f7)",
    button_primary_text_color="#ffffff",
)

CSS = """
#hdr h1 { margin: 0; font-size: 1.6rem; }
.pill { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.75rem;
        font-weight:600; margin-right:6px; }
.pill-amber { background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.4); }
.pill-warn  { background:rgba(239,68,68,0.12);  color:#fca5a5; border:1px solid rgba(239,68,68,0.35); }
#badge textarea { font-family: var(--font-mono) !important; font-size:12px !important;
                  background:#0d1117 !important; color:#9fe0ff !important; }
.prov textarea { font-family: var(--font-mono) !important; font-size:12px !important; }
.links a { margin-right:14px; text-decoration:none; font-weight:600; color:#a5b4fc; }
.links a:hover { color:#c4b5fd; }
footer { display:none !important; }
@media (max-width: 860px) {
  #workbench { flex-direction: column-reverse !important; }
}
"""


# --------------------------------------------------------------------------- #
# Build UI
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="WaveDiT - 3D Brain MRI Generator", theme=THEME, css=CSS) as demo:

        # ---- Header ----
        with gr.Row(elem_id="hdr"):
            gr.Markdown(
                "# \U0001f9e0 WaveDiT\n"
                "Generate and explore synthetic 3D brain MRI, conditioned on age, in your browser.\n\n"
                "**Danilo Danese**, Angela Lombardi, Giuseppe Fasano, Matteo Attimonelli, "
                "Tommaso Di Noia at SisInfLab, Politecnico di Bari."
            )
        gr.HTML(
            '<div>'
            '<span class="pill pill-amber">Pre-release checkpoints</span>'
            '<span class="pill pill-warn">&#9888;&#65039; Research demo, synthetic, not for clinical use</span>'
            '</div>'
            f'<div class="links" style="margin-top:8px">'
            f'<a href="{LINK_PAPER}" target="_blank">\U0001f4c4 Paper</a>'
            f'<a href="{LINK_CODE}" target="_blank">\U0001f4bb Code</a>'
            f'<a href="{LINK_PROJECT}" target="_blank">\U0001f310 Project page</a>'
            f'<a href="{LINK_MODEL}" target="_blank">\U0001f917 Model</a>'
            '</div>'
        )

        sweep_state = gr.State({})

        # ---- Workbench ----
        with gr.Row(elem_id="workbench"):
            # LEFT: controls
            with gr.Column(scale=38):
                with gr.Group():
                    gr.Markdown("### Essentials")
                    with gr.Row():
                        age = gr.Slider(AGE_MIN, AGE_MAX, value=AGE_DEFAULT, step=1,
                                        label="Age (years)", scale=8)
                        age_rand = gr.Button("\U0001f3b2", scale=1, min_width=44,
                                             elem_id="age-rand")
                    age_help = gr.Markdown(age_microcopy(AGE_DEFAULT))
                    model = gr.Radio(
                        list(CHECKPOINTS.keys()), value=DEFAULT_MODEL, label="Model",
                        info="Base = faster, ~3 GB VRAM. FinePatch = sharper, slower.",
                    )
                    generate_btn = gr.Button("✦ Generate brain", variant="primary", size="lg")
                    status = gr.Markdown("Ready. Pick an age and press Generate.")

                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        seed = gr.Number(value=SEED_DEFAULT, precision=0, label="Seed", scale=8)
                        seed_rand = gr.Button("\U0001f3b2", scale=1, min_width=44,
                                              elem_id="seed-rand")
                    steps = gr.Slider(STEPS_MIN, STEPS_MAX, value=STEPS_DEFAULT, step=1,
                                      label="ODE steps")
                    steps_help = gr.Markdown(steps_microcopy(STEPS_DEFAULT))
                    cfg_scale = gr.Slider(CFG_MIN, CFG_MAX, value=CFG_DEFAULT, step=0.1,
                                          label="CFG scale")
                    cfg_help = gr.Markdown(cfg_microcopy(CFG_DEFAULT))
                    sampler = gr.Radio(SAMPLERS, value="Heun", label="Sampler",
                                       info="Heun = 2nd-order (2 evals/step); Euler = 1st-order, faster.")
                    morpheus = gr.Slider(0.0, 2.0, value=MORPHEUS_DEFAULT, step=0.05,
                                         label="Morpheus uncertainty guidance",
                                         info="Descends predicted per-band uncertainty, peaks at t=0.5 "
                                              "(prop. sin(pi*t)). 0 = ablate; 1.0 = trained default.")

                with gr.Accordion("Presets", open=False):
                    with gr.Row():
                        p_child = gr.Button("\U0001f331 Child (8)", size="sm")
                        p_prime = gr.Button("\U0001f9d1 Prime years (35)", size="sm")
                        p_elder = gr.Button("\U0001f333 Wise elder (82)", size="sm")
                    with gr.Row():
                        p_show = gr.Button("\U0001f48e Showcase (FinePatch, 80 steps)", size="sm")
                        p_abl = gr.Button("\U0001f52c Ablation: Morpheus off", size="sm")
                        p_surprise = gr.Button("\U0001f3b2 Surprise me", size="sm")

                with gr.Accordion("Aging time-lapse", open=False):
                    gr.Markdown(
                        "Sweep age across frames and scrub the result in the same viewer. "
                        f"To stay within the GPU budget, each frame uses at most "
                        f"{SWEEP_STEPS_MAX} ODE steps."
                    )
                    with gr.Row():
                        start_age = gr.Slider(AGE_MIN, AGE_MAX, value=max(AGE_MIN, 20), step=1,
                                              label="Start age")
                        end_age = gr.Slider(AGE_MIN, AGE_MAX, value=min(AGE_MAX, 80), step=1,
                                            label="End age")
                    with gr.Row():
                        frames = gr.Slider(SWEEP_FRAMES_MIN, SWEEP_FRAMES_MAX, value=7, step=1,
                                           label="Frames")
                        fix_seed = gr.Checkbox(value=True, label="Fix seed across ages")
                    sweep_btn = gr.Button("▶ Run aging sweep", variant="secondary")
                    frame_slider = gr.Slider(0, SWEEP_FRAMES_MAX - 1, value=0, step=1,
                                             visible=False, label="Time-lapse frame")

            # RIGHT: viewer
            with gr.Column(scale=62):
                viewer = gr.HTML(_initial_viewer_html())
                with gr.Row():
                    colormap = gr.Dropdown(COLORMAPS, value="gray", label="Colormap", scale=3)
                    gamma = gr.Slider(0.4, 2.0, value=1.0, step=0.05, label="Brightness (gamma)", scale=3)
                    reset_view = gr.Button("Reset view", scale=1)
                badge = gr.Textbox(
                    value="(generate a brain to see timing + VRAM)",
                    label="Generation report", interactive=False, lines=2, elem_id="badge",
                )
                caption = gr.Textbox(
                    value="Provenance will appear here after you generate.",
                    label="Provenance (copy for your records)", interactive=False, lines=1,
                    elem_classes=["prov"],
                )
                download = gr.DownloadButton("⬇ Download .nii.gz (float32)", visible=False)
                gr.Markdown(
                    "Left-drag the 3D pane to rotate and cut into the brain &middot; scroll to zoom "
                    "&middot; right-drag for contrast. The three flat panels are axial, coronal, sagittal."
                )
                gallery = gr.Gallery(label="Aging time-lapse - mid-axial slices",
                                     visible=False, columns=8, height=160, object_fit="contain")

        # ---- Accordions ----
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "WaveDiT learns a velocity field in the 3D Haar-wavelet domain: a forward DWT splits "
                "each 224 cubed volume into 8 frequency bands, a flow-matching ODE is integrated in that "
                "compact space, and an inverse DWT reconstructs the volume - making full-resolution "
                "3D generation tractable on one GPU. You steer it with one condition: **age**. CFG uses "
                "std-rescaling (`cfg_rescale=0.7`), inactive at CFG=1. Neighborhood attention runs in a "
                "pure-PyTorch fallback (`WAVEDIT_NA_BACKEND=torch`), numerically equivalent to the NATTEN "
                "kernels used for training."
            )

        with gr.Accordion("Cite this work", open=False):
            gr.Code(value=CITATION, language=None, label="BibTeX")

        with gr.Accordion("FAQ", open=False):
            gr.Markdown(
                "**Why does the very first request after a cold start take longer?** ZeroGPU spins up "
                "the GPU and attaches it on demand; both checkpoints are loaded once at startup and stay "
                "resident, so subsequent runs only pay for sampling.\n\n"
                "**Why are some runs slower?** FinePatch and high step counts do more work; the Heun "
                "sampler runs two network evaluations per step.\n\n"
                "**The 3D viewer is blank.** It needs WebGL2. Use a recent Chrome/Firefox/Safari, or "
                "download the `.nii.gz` and open it in your own viewer."
            )

        # ---- Footer ----
        gr.HTML(
            '<div style="margin-top:18px;padding-top:12px;border-top:1px solid #232a39;'
            'color:#7c8aa0;font-size:0.8rem;line-height:1.6">'
            'WaveDiT is a research artifact (MICCAI 2026). Synthetic data only - not a medical device, '
            'not for clinical use. &copy; The authors &middot; CC-BY-NC-4.0.<br>'
            f'MICCAI 2026 &middot; arXiv:2606.08670 &middot; space v{SPACE_VERSION}'
            '</div>'
        )

        # ----------------------------------------------------------------- #
        # Wiring
        # ----------------------------------------------------------------- #
        gen_inputs = [model, age, seed, steps, cfg_scale, sampler, morpheus, colormap]
        gen_outputs = [viewer, status, badge, caption, download]

        generate_btn.click(generate, inputs=gen_inputs, outputs=gen_outputs)

        # Living microcopy
        age.change(age_microcopy, inputs=age, outputs=age_help)
        cfg_scale.change(cfg_microcopy, inputs=cfg_scale, outputs=cfg_help)
        steps.change(steps_microcopy, inputs=[steps, model], outputs=steps_help)
        model.change(steps_microcopy, inputs=[steps, model], outputs=steps_help)

        # Random buttons (no auto-generate)
        age_rand.click(random_age, outputs=age)
        seed_rand.click(random_seed, outputs=seed)

        # Live viewer updates via postMessage (no regeneration) to the existing iframe.
        colormap.change(
            None, inputs=colormap, outputs=None,
            js="(c) => { const f=document.querySelector('#workbench iframe'); "
               "if (f && f.contentWindow) f.contentWindow.postMessage({type:'colormap',value:c},'*'); }",
        )
        gamma.change(
            None, inputs=gamma, outputs=None,
            js="(g) => { const f=document.querySelector('#workbench iframe'); "
               "if (f && f.contentWindow) f.contentWindow.postMessage({type:'gamma',value:g},'*'); }",
        )
        reset_view.click(
            None, inputs=None, outputs=None,
            js="() => { const f=document.querySelector('#workbench iframe'); "
               "if (f && f.contentWindow) f.contentWindow.postMessage({type:'reset'},'*'); }",
        )

        # Presets (set controls, refresh age microcopy, then auto-generate)
        preset_targets = [model, age, seed, steps, cfg_scale, sampler, morpheus, colormap]
        for btn, fn in [
            (p_child, preset_child), (p_prime, preset_prime), (p_elder, preset_elder),
            (p_show, preset_showcase), (p_abl, preset_ablation), (p_surprise, preset_surprise),
        ]:
            btn.click(fn, outputs=preset_targets).then(
                age_microcopy, inputs=age, outputs=age_help
            ).then(generate, inputs=gen_inputs, outputs=gen_outputs)

        # Aging time-lapse
        sweep_inputs = [model, start_age, end_age, frames, fix_seed, seed,
                        steps, cfg_scale, sampler, morpheus, colormap]
        sweep_btn.click(
            run_sweep, inputs=sweep_inputs,
            outputs=[viewer, gallery, status, frame_slider, sweep_state],
        )
        frame_slider.change(show_sweep_frame, inputs=[frame_slider, sweep_state], outputs=viewer)

    return demo


demo = build_demo()

if __name__ == "__main__":
    # allowed_paths lets Gradio serve the float32 .nii.gz download from /tmp.
    # WAVEDIT_SHARE=1 opens a public *.gradio.live tunnel (handy when the server is
    # remote and localhost isn't reachable, e.g. VS Code over VPN). Off by default.
    demo.queue(max_size=24).launch(
        allowed_paths=[str(DOWNLOAD_DIR)],
        share=os.environ.get("WAVEDIT_SHARE") == "1",
    )
