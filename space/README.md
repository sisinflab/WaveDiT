---
title: WaveDiT — 3D Brain MRI Generator
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Generate & explore synthetic 3D brain MRIs in your browser (research only)
models:
  - danesed/WaveDiT
disable_embedding: false
---

# 🧠 WaveDiT — 3D Brain MRI Generator

Generate and explore **synthetic 3D brain MRI**, conditioned on age — entirely in your browser.

WaveDiT learns a velocity field in the **3D Haar-wavelet domain**: a forward DWT splits each
224³ volume into 8 frequency bands, a flow-matching ODE is integrated in that compact latent,
and an inverse DWT reconstructs the full-resolution volume — making 3D generation tractable on
a single GPU. You steer it with one condition: **age** (6–95). This Space runs on
**ZeroGPU** and renders results with an MRIcroGL-like **Niivue** viewer (axial / coronal /
sagittal panels + an interactive 3D render you can rotate and cut into with a clip plane).

> ⚠️ **Research demo — pre-release weights.** Synthetic images only. **Not a medical device,
> not for clinical or diagnostic use.**

## Features

- **One-click generation** — pick an age, press *Generate*, get a 3D brain you can rotate & slice.
- **Two models** — *Base* (fast, ~3 GB VRAM) and *FinePatch* (sharper, slower). Both 142M params.
- **Full control** — seed, ODE steps (1–200, default 10), CFG scale, sampler (Heun/Euler), and
  *Morpheus* state-aware uncertainty guidance (set to 0 to ablate).
- **🎲 Random** buttons for age and seed, plus curated **presets** (Child / Prime / Elder /
  Showcase / Ablation / Surprise me).
- **Aging time-lapse** — fix the seed and sweep age across up to 8 frames (≤50 steps/frame to stay
  within the GPU budget); scrub the result in the same viewer and view a labelled mid-axial montage.
  Qualitative illustration, not a biomarker.
- **Live viewer controls** — colormap and gamma update instantly via `postMessage` (no regen).
- **Reproducibility instruments**: always-on generation/VRAM badge, copyable provenance caption,
  and a float32 `.nii.gz` download.

## How it runs (ZeroGPU)

- The `wavedit/` package is **vendored** into this Space (the repo's pyproject pins
  `torch==2.6.0`, which is unsupported on ZeroGPU ≥ 2.8 — so we do not pip-install it). A lazy
  `import wandb` patch lets the training-only logging dependency be dropped, and the never-used
  `evaluation/` and `data/` subtrees are pruned from the vendored copy.
- `WAVEDIT_NA_BACKEND=torch` is set before importing wavedit (NATTEN is unavailable on Spaces;
  the pure-PyTorch neighborhood-attention fallback is numerically equivalent).
- **Both checkpoints are loaded onto `cuda` once at module-scope startup** (the ZeroGPU contract:
  `@spaces.GPU` calls run in a forked subprocess, so any cuda placement / cache built *inside* the
  decorated function would be discarded after each call and silently rebuilt — wasting the user's
  limited GPU quota). Inside the GPU function we only set the ODE-step count in place (a mutable
  attribute, no rebuild) and call `model.sample()`.
- Single generation runs inside `@spaces.GPU(duration=…)` with a dynamic duration that scales with
  model + steps. The **aging time-lapse runs each frame as its own short GPU call** (at most 50
  steps/frame, up to 8 frames) and drives the progress bar from the main process — so no single GPU
  window can overrun its declared budget (a long multi-frame run in one call would be killed midway).
- Checkpoints are downloaded from [`danesed/WaveDiT`](https://huggingface.co/danesed/WaveDiT).

**Provisioning:** ZeroGPU is selected in **Space Settings → Hardware** (needs a PRO/Team/Enterprise
account that owns the Space). There is no README YAML key that enforces ZeroGPU. `large` (48 GB) is
sufficient — resident Base (~3 GB) + FinePatch (~8.4 GB peak) fit comfortably; do **not** request
`xlarge` (it doubles the quota cost).

## Links

- 📄 Paper: https://arxiv.org/abs/2606.08670
- 💻 Code: https://github.com/sisinflab/WaveDiT
- 🌐 Project page: https://danesed.github.io/wavedit-page/
- 🤗 Model: https://huggingface.co/danesed/WaveDiT

## Citation

```bibtex
@misc{danese2026waveditdistributionawarewaveletflow,
  title         = {WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis},
  author        = {Danilo Danese and Angela Lombardi and Giuseppe Fasano and Matteo Attimonelli and Tommaso Di Noia},
  year          = {2026},
  eprint        = {2606.08670},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.08670}
}
```

## License & disclaimer

Code & weights: **CC-BY-NC-4.0**. WaveDiT is a research artifact (MICCAI 2026). Synthetic data
only — not a medical device, not for clinical use. © The authors.
