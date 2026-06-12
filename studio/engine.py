"""Generation engine for WaveDiT Studio.

Owns the device policy, the lazy per-checkpoint model registry, and the single-worker
job queue for generations and aging sweeps. Progress is measured in real backbone
forward calls (NFE) via a forward hook and streamed to the UI through the publish
callback supplied by the server. Research artifact only: synthetic images, not a
medical device, not for clinical use.
"""

from __future__ import annotations

import os

# Silence the tqdm bar inside wavedit's ODE integrator (must precede wavedit imports).
os.environ.setdefault("TQDM_DISABLE", "1")

import re
import secrets
import threading
import time
from typing import Any, Callable

import numpy as np
import torch

from wavedit import Config
from wavedit.models import build_model
from wavedit.training.checkpoint import load_model_weights
from wavedit.generation.generator import center_crop_bounds

from .library import library
from .paths import checkpoints_dir
from .settings import settings
from .sysinfo import get_sysinfo, pick_device

Publish = Callable[[str, dict], None]

# Output grid: every checkpoint is center-cropped to the MNI-like research grid.
CROP_SIZE = (182, 218, 182)
DEFAULT_FULL_SIZE = (224, 224, 224)

AGE_FALLBACK = (6, 95)
SEED_DEFAULT = 42
SEED_MAX = 2_147_483_647
STEPS_MIN, STEPS_MAX = 1, 200
CFG_MIN, CFG_MAX = 1.0, 8.0
MORPHEUS_MIN, MORPHEUS_MAX = 0.0, 2.0
CFG_RESCALE = 0.7  # fixed, only active when cfg_scale != 1.0 (mirrors the Space)
FRAMES_MIN, FRAMES_MAX = 2, 60

PROGRESS_MIN_INTERVAL_S = 0.150  # throttle for gen_progress / sweep_progress events
ETA_EMA_ALPHA = 0.3              # EMA weight of the newest inter-call wall time
CALIBRATION_EMA_ALPHA = 0.3      # EMA weight of the newest sec-per-NFE measurement
MPS_SAMPLE_INTERVAL_S = 0.2      # memory monitor sampling period on Apple MPS

_SAFE_CKPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pth$")


class EngineBusy(RuntimeError):
    """Raised when a generation or sweep is requested while another job is running."""


class GenerationCancelled(Exception):
    """Raised from the progress hook (or between sweep frames) when a job is cancelled."""


def _label_for(filename: str) -> str:
    """Human label for a checkpoint file, e.g. WaveDiT-Base.pth -> Base."""
    stem = filename[:-4] if filename.endswith(".pth") else filename
    if stem.startswith("WaveDiT-"):
        stem = stem[len("WaveDiT-"):]
    return stem or filename


def _coerce_seed(seed: Any) -> int:
    try:
        return abs(int(seed)) % (SEED_MAX + 1)
    except (TypeError, ValueError):
        return SEED_DEFAULT


def _coerce_float(value: Any, default: float, lo: float, hi: float, ndigits: int = 2) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if out != out:  # NaN guard
        out = default
    return round(min(hi, max(lo, out)), ndigits)


def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return min(hi, max(lo, out))


def _nfe_total(steps: int, sampler: str, cfg_scale: float) -> int:
    """Total backbone forward calls for one sample() run (the progress denominator)."""
    velocity_calls = 2 * steps - 1 if sampler == "heun" else steps
    return velocity_calls * (2 if cfg_scale != 1.0 else 1)


class _MemoryMonitor:
    """Peak accelerator memory during a job: native counters on cuda, sampling on mps."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        elif self._device.type == "mps":
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                current = int(torch.mps.current_allocated_memory())
            except Exception:
                current = 0
            self._peak_bytes = max(self._peak_bytes, current)
            self._stop.wait(MPS_SAMPLE_INTERVAL_S)

    def stop(self) -> float:
        """Stop sampling and return the observed peak in GiB (0.0 on cpu)."""
        if self._device.type == "cuda":
            try:
                return float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
            except Exception:
                return 0.0
        if self._device.type == "mps":
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            return self._peak_bytes / (1024 ** 3)
        return 0.0


class Engine:
    """Single-worker generation engine with a lazy, device-resident model registry."""

    def __init__(self) -> None:
        self._job_lock = threading.Lock()        # at most one generation/sweep at a time
        self._registry_lock = threading.Lock()   # protects registry bookkeeping
        self._registry: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, threading.Event] = {}
        self._device: torch.device | None = None
        self._last_loaded: str | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        return {
            "device": get_sysinfo(),
            "busy": self.busy(),
            "calibration": dict(settings.get().get("calibration") or {}),
            "age_range": list(self.age_range()),
            "mem_now_gb": round(self._mem_used_gb(), 2),
        }

    def _mem_used_gb(self) -> float:
        """Accelerator memory currently in use (GiB); 0.0 on cpu or on any error."""
        device = self._device or pick_device()
        try:
            if device.type == "cuda":
                return float(torch.cuda.memory_allocated()) / (1024 ** 3)
            if device.type == "mps":
                return float(torch.mps.current_allocated_memory()) / (1024 ** 3)
        except Exception:
            return 0.0
        return 0.0

    def validate_checkpoint(self, filename: str) -> str:
        """Confirm a checkpoint is a loadable WaveDiT model; return its label.

        Raises ValueError with a human message if the file is not a WaveDiT
        checkpoint, so the import flow can reject it before it reaches the picker.
        """
        path = checkpoints_dir() / filename
        if not path.is_file():
            raise ValueError(f"{filename} is not in the checkpoints directory")
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not read the checkpoint ({exc})") from exc
        required = ("config", "condition_config", "condition_ranges",
                    "categorical_maps", "null_conditions")
        missing = [k for k in required if k not in ck]
        if missing:
            raise ValueError(
                "this file is not a WaveDiT checkpoint "
                f"(missing: {', '.join(missing)})"
            )
        try:
            Config.from_dict(ck["config"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"unsupported checkpoint config ({exc})") from exc
        return _label_for(filename)

    def age_range(self) -> tuple[int, int]:
        """Age bounds from the most recently loaded checkpoint, fallback (6, 95)."""
        with self._registry_lock:
            entry = self._registry.get(self._last_loaded) if self._last_loaded else None
        if entry is None:
            return AGE_FALLBACK
        return entry["age_lo"], entry["age_hi"]

    def busy(self) -> bool:
        return self._job_lock.locked()

    def cancel(self, job_id: str) -> bool:
        event = self._jobs.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def generate(self, params: dict, publish: Publish) -> str:
        return self._spawn(self._run_generate, params, publish)

    def sweep(self, params: dict, publish: Publish) -> str:
        return self._spawn(self._run_sweep, params, publish)

    # ------------------------------------------------------------------ #
    # Job plumbing
    # ------------------------------------------------------------------ #
    def _spawn(self, target: Callable, params: dict, publish: Publish) -> str:
        if not self._job_lock.acquire(blocking=False):
            raise EngineBusy("Another generation is already running.")
        job_id = f"job-{secrets.token_hex(6)}"
        cancel_ev = threading.Event()
        self._jobs[job_id] = cancel_ev
        try:
            thread = threading.Thread(
                target=target,
                args=(job_id, dict(params or {}), publish, cancel_ev),
                daemon=True,
                name=f"wavedit-{job_id}",
            )
            thread.start()
        except Exception:
            self._jobs.pop(job_id, None)
            self._job_lock.release()
            raise
        return job_id

    def _run_generate(self, job_id: str, params: dict, publish: Publish, cancel_ev: threading.Event) -> None:
        try:
            p = self._validate_common(params)
            entry = self._entry(p["model"])
            p["age"] = self._clamp_age(params.get("age"), entry)
            nfe_total = _nfe_total(p["steps"], p["sampler"], p["cfg_scale"])
            _safe(publish, "gen_start", {"job_id": job_id, "nfe_total": nfe_total, "params": p})

            def progress(done: int, total: int, ema: float, elapsed: float) -> None:
                _safe(publish, "gen_progress", {
                    "job_id": job_id,
                    "nfe_done": done,
                    "nfe_total": total,
                    "pct": round(100.0 * done / total, 1),
                    "eta_s": round(max(0.0, ema * (total - done)), 1),
                    "elapsed_s": round(elapsed, 1),
                    "mem_gb": round(self._mem_used_gb(), 2),
                })

            item = self._generate_one(p, entry, nfe_total, cancel_ev, sweep_id=None, progress_cb=progress)
            _safe(publish, "gen_done", {"job_id": job_id, "item": item})
        except GenerationCancelled:
            _safe(publish, "gen_error", {"job_id": job_id, "message": "Generation cancelled.", "cancelled": True})
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            _safe(publish, "gen_error", {"job_id": job_id, "message": str(exc) or type(exc).__name__, "cancelled": False})
        finally:
            self._jobs.pop(job_id, None)
            self._job_lock.release()

    def _run_sweep(self, job_id: str, params: dict, publish: Publish, cancel_ev: threading.Event) -> None:
        try:
            p = self._validate_common(params)
            entry = self._entry(p["model"])
            frames = _coerce_int(params.get("frames"), 5, FRAMES_MIN, FRAMES_MAX)
            fix_seed = bool(params.get("fix_seed", True))
            age_start = self._clamp_age(params.get("age_start"), entry)
            age_end = self._clamp_age(params.get("age_end"), entry)
            ages = [round(float(a), 1) for a in np.linspace(age_start, age_end, frames)]
            nfe_per_frame = _nfe_total(p["steps"], p["sampler"], p["cfg_scale"])
            sweep_id = f"sweep-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"

            sweep_params = {**p, "age_start": age_start, "age_end": age_end,
                            "frames": frames, "fix_seed": fix_seed}
            _safe(publish, "sweep_start", {"job_id": job_id, "frames": frames, "params": sweep_params})

            ids: list[str] = []
            for i, age in enumerate(ages):
                if cancel_ev.is_set():
                    raise GenerationCancelled()
                frame_seed = p["seed"] if fix_seed else (p["seed"] + i) % (SEED_MAX + 1)

                def progress(done: int, total: int, ema: float, elapsed: float, _i: int = i) -> None:
                    overall = (_i + done / total) / frames
                    remaining_calls = (total - done) + (frames - _i - 1) * total
                    _safe(publish, "sweep_progress", {
                        "job_id": job_id,
                        "frame": _i,
                        "frames": frames,
                        "pct": round(100.0 * overall, 1),
                        "eta_s": round(max(0.0, ema * remaining_calls), 1),
                        "mem_gb": round(self._mem_used_gb(), 2),
                    })

                frame_params = {**p, "age": ages[i], "seed": frame_seed}
                item = self._generate_one(frame_params, entry, nfe_per_frame, cancel_ev,
                                          sweep_id=sweep_id, progress_cb=progress)
                ids.append(item["id"])
                _safe(publish, "sweep_frame_done", {"job_id": job_id, "frame": i, "item": item})

            _safe(publish, "sweep_done", {"job_id": job_id, "sweep_id": sweep_id, "ids": ids})
        except GenerationCancelled:
            _safe(publish, "gen_error", {"job_id": job_id, "message": "Sweep cancelled.", "cancelled": True})
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            _safe(publish, "gen_error", {"job_id": job_id, "message": str(exc) or type(exc).__name__, "cancelled": False})
        finally:
            self._jobs.pop(job_id, None)
            self._job_lock.release()

    # ------------------------------------------------------------------ #
    # One generation (validated params -> saved library item)
    # ------------------------------------------------------------------ #
    def _generate_one(
        self,
        p: dict,
        entry: dict,
        nfe_total: int,
        cancel_ev: threading.Event,
        sweep_id: str | None,
        progress_cb: Callable[[int, int, float, float], None],
    ) -> dict:
        device = self._get_device()
        precision = self._resolve_precision(device)
        arr, wall_s, peak_gb, used_precision = self._sample(
            entry, p, nfe_total, precision, cancel_ev, progress_cb
        )
        self._update_calibration(p["model"], used_precision, device, wall_s, nfe_total)
        meta = {
            "model": p["model"],
            "model_label": _label_for(p["model"]),
            "age": p["age"],
            "seed": p["seed"],
            "steps": p["steps"],
            "cfg_scale": p["cfg_scale"],
            "sampler": p["sampler"],
            "morpheus": p["morpheus"],
            "precision": used_precision,
            "wall_s": round(wall_s, 2),
            "nfe_total": nfe_total,
            "peak_mem_gb": round(peak_gb, 2),
            "device": device.type,
            "sweep_id": sweep_id,
        }
        return library.save(arr, meta)

    def _sample(
        self,
        entry: dict,
        p: dict,
        nfe_total: int,
        precision: str,
        cancel_ev: threading.Event,
        progress_cb: Callable[[int, int, float, float], None],
    ) -> tuple[np.ndarray, float, float, str]:
        """Run sample() with progress hook and memory monitor; bf16 falls back to float32."""
        attempts = ["bf16", "float32"] if precision == "bf16" else ["float32"]
        last_index = len(attempts) - 1
        for idx, attempt in enumerate(attempts):
            if cancel_ev.is_set():
                raise GenerationCancelled()
            try:
                arr, wall_s, peak_gb = self._sample_attempt(entry, p, nfe_total, attempt, cancel_ev, progress_cb)
            except GenerationCancelled:
                raise
            except Exception:
                if attempt == "bf16" and idx < last_index:
                    settings.update({"bf16_ok": False})
                    continue
                raise
            if attempt == "bf16" and settings.get().get("bf16_ok") is not True:
                settings.update({"bf16_ok": True})
            return arr, wall_s, peak_gb, attempt
        raise RuntimeError("Sampling failed for every precision attempt.")  # unreachable

    def _sample_attempt(
        self,
        entry: dict,
        p: dict,
        nfe_total: int,
        precision: str,
        cancel_ev: threading.Event,
        progress_cb: Callable[[int, int, float, float], None],
    ) -> tuple[np.ndarray, float, float]:
        device = self._get_device()
        model = entry["model"]
        model.num_flow_steps_sampling = p["steps"]

        torch.manual_seed(p["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(p["seed"])

        t0 = time.perf_counter()
        hook_state = {"count": 0, "last_pub": 0.0, "last_t": t0, "ema": None}

        def _hook(_module, _inputs, _output) -> None:
            if cancel_ev.is_set():
                raise GenerationCancelled()
            now = time.perf_counter()
            hook_state["count"] += 1
            dt = now - hook_state["last_t"]
            hook_state["last_t"] = now
            ema = hook_state["ema"]
            ema = dt if ema is None else ETA_EMA_ALPHA * dt + (1.0 - ETA_EMA_ALPHA) * ema
            hook_state["ema"] = ema
            done = hook_state["count"]
            final = done >= nfe_total
            if final or (now - hook_state["last_pub"]) >= PROGRESS_MIN_INTERVAL_S:
                hook_state["last_pub"] = now
                progress_cb(min(done, nfe_total), nfe_total, ema, now - t0)

        monitor = _MemoryMonitor(device)
        handle = model.backbone.register_forward_hook(_hook)
        monitor.start()
        try:
            morpheus = p["morpheus"]
            with torch.no_grad():
                vol = model.sample(
                    num_samples=1,
                    raw_conditions={"age": float(p["age"])},
                    cfg_scale=float(p["cfg_scale"]),
                    sampler=p["sampler"],
                    morpheus_scale=None if abs(morpheus - 1.0) < 1e-6 else morpheus,
                    cfg_rescale=CFG_RESCALE,
                    autocast_dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
                )
            self._synchronize(device)
            wall_s = time.perf_counter() - t0
        finally:
            handle.remove()
            peak_gb = monitor.stop()

        arr = self._postprocess(vol, entry["full_size"])
        return arr, wall_s, peak_gb

    @staticmethod
    def _postprocess(vol: torch.Tensor, full_size: tuple[int, int, int]) -> np.ndarray:
        """(1,1,D,H,W) in [-1,1] -> (182,218,182) float32 in [0,1] (CPU numpy)."""
        vol = torch.clamp((vol.float() + 1.0) / 2.0, 0.0, 1.0)
        (d0, d1), (h0, h1), (w0, w1) = center_crop_bounds(full_size, CROP_SIZE)
        vol = vol[:, :, d0:d1, h0:h1, w0:w1]
        return vol[0, 0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Model registry
    # ------------------------------------------------------------------ #
    def _get_device(self) -> torch.device:
        if self._device is None:
            self._device = pick_device()
        return self._device

    def _entry(self, filename: str) -> dict:
        """Cached registry entry for a checkpoint file; builds and loads it on first use."""
        with self._registry_lock:
            entry = self._registry.get(filename)
            if entry is not None:
                self._last_loaded = filename
                return entry

        path = checkpoints_dir() / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint '{filename}' is not downloaded. Download it from the Models panel first."
            )

        device = self._get_device()
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = Config.from_dict(ck["config"])
        model = build_model(
            cfg,
            ck["condition_config"],
            ck["condition_ranges"],
            ck["categorical_maps"],
            ck["null_conditions"],
        )
        load_model_weights(model, ck)
        model.to(device).eval()

        try:
            full_size = tuple(int(s) for s in cfg.data.image_size)
        except Exception:
            full_size = DEFAULT_FULL_SIZE

        age_lo, age_hi = AGE_FALLBACK
        try:
            rng = ck.get("condition_ranges", {}).get("age", {})
            lo = int(round(float(rng.get("min", AGE_FALLBACK[0]))))
            hi = int(round(float(rng.get("max", AGE_FALLBACK[1]))))
            if hi > lo:
                age_lo, age_hi = lo, hi
        except Exception:
            pass

        entry = {"model": model, "full_size": full_size, "age_lo": age_lo, "age_hi": age_hi}
        with self._registry_lock:
            self._registry[filename] = entry
            self._last_loaded = filename
        return entry

    # ------------------------------------------------------------------ #
    # Validation, precision, calibration
    # ------------------------------------------------------------------ #
    def _validate_common(self, params: dict) -> dict:
        filename = str(params.get("model") or settings.get().get("default_model") or "WaveDiT-Base.pth")
        if not _SAFE_CKPT_RE.match(filename) or ".." in filename:
            raise ValueError(f"Invalid model filename: {filename!r}")
        sampler = str(params.get("sampler") or "heun").strip().lower()
        if sampler not in ("heun", "euler"):
            sampler = "heun"
        return {
            "model": filename,
            "seed": _coerce_seed(params.get("seed")),
            "steps": _coerce_int(params.get("steps"), 10, STEPS_MIN, STEPS_MAX),
            "cfg_scale": _coerce_float(params.get("cfg_scale"), 1.0, CFG_MIN, CFG_MAX),
            "sampler": sampler,
            "morpheus": _coerce_float(params.get("morpheus"), 1.0, MORPHEUS_MIN, MORPHEUS_MAX),
        }

    @staticmethod
    def _clamp_age(value: Any, entry: dict) -> float:
        lo, hi = float(entry["age_lo"]), float(entry["age_hi"])
        default = round((lo + hi) / 2.0, 1)
        return _coerce_float(value, default, lo, hi, ndigits=1)

    def _resolve_precision(self, device: torch.device) -> str:
        """Settings-driven precision: "auto" probes bf16 once, results are persisted."""
        if device.type == "cpu":
            return "float32"  # autocast is disabled on cpu in the vendored sample()
        if device.type == "mps":
            # On macOS < 14 the mps autocast constructor silently disables itself for
            # bf16 (warning, no exception), so the probe below would "succeed" while
            # actually running float32: gate on the same condition torch uses.
            try:
                if not torch.backends.mps.is_macos_or_newer(14, 0):
                    return "float32"
            except AttributeError:
                pass
        current = settings.get()
        mode = current.get("precision", "auto")
        if mode == "bf16":
            return "bf16"
        if mode == "float32":
            return "float32"
        bf16_ok = current.get("bf16_ok")
        if bf16_ok is True:
            return "bf16"
        if bf16_ok is False:
            return "float32"
        return "bf16"  # unknown: probe once; _sample() persists the outcome

    def _update_calibration(self, filename: str, precision: str, device: torch.device,
                            wall_s: float, nfe_total: int) -> None:
        """Persist an EMA of seconds per NFE so the UI can show live time estimates."""
        key = f"{filename}|{precision}|{device.type}"
        sec_per_nfe = wall_s / max(1, nfe_total)
        calibration = dict(settings.get().get("calibration") or {})
        prev = calibration.get(key)
        if isinstance(prev, (int, float)) and prev > 0:
            sec_per_nfe = CALIBRATION_EMA_ALPHA * sec_per_nfe + (1.0 - CALIBRATION_EMA_ALPHA) * float(prev)
        calibration[key] = round(sec_per_nfe, 4)
        settings.update({"calibration": calibration})


def _safe(publish: Publish, event: str, data: dict) -> None:
    """Publish an SSE event without letting a UI/transport hiccup abort the job."""
    try:
        publish(event, data)
    except Exception:
        pass


engine = Engine()
