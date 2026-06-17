---
license: mit
library_name: pytorch
pipeline_tag: unconditional-image-generation
tags:
  - medical-imaging
  - mri
  - brain
  - neuroimaging
  - 3d
  - diffusion
  - flow-matching
  - generative
---

# WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis

[![Star on GitHub](https://img.shields.io/github/stars/sisinflab/WaveDiT?style=social)](https://github.com/sisinflab/WaveDiT)

[![WaveDiT Studio for macOS](https://img.shields.io/badge/WaveDiT%20Studio-macOS%20app-000000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/sisinflab/WaveDiT/tree/macos-app)

WaveDiT synthesises full-resolution, age-conditioned 3D brain MRIs by performing
**conditional flow matching in the 3D Haar wavelet domain** with a slice-wise **HDiT**
transformer backbone, guided by **Morpheus**, a state-aware uncertainty scheduler that
adaptively weights the loss and sampling across frequency bands.

> 🤗 **Try it live, no install:** pick an age and generate a synthetic 3D brain MRI you can
> rotate and slice in your browser &rarr; **[Demo Space](https://huggingface.co/spaces/danesed/WaveDiT-demo)**


Official model release for the MICCAI 2026 paper:

> **WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis**
> Danilo Danese, Angela Lombardi, Giuseppe Fasano, Matteo Attimonelli, Tommaso Di Noia
> [arXiv:2606.08670](https://arxiv.org/abs/2606.08670)

**Links:** [🤗 Live demo](https://huggingface.co/spaces/danesed/WaveDiT-demo) ·
[Code (GitHub)](https://github.com/sisinflab/WaveDiT) ·
[Project page](https://danesed.github.io/wavedit-page/) ·
[HF paper](https://huggingface.co/papers/2606.08670) ·
[arXiv](https://arxiv.org/abs/2606.08670) ·


<p align="center">
  <img src="assets/WaveDiT_Architecture.png" width="760" alt="WaveDiT architecture">
</p>

## Model description

- **Wavelets:** one-level 3D Haar wavelet transform of a 224³ T1-weighted volume →
  an 8-channel 112³ representation (1 low-frequency LLL band + 7 high-frequency bands).
- **Objective:** conditional flow matching (linear interpolant, velocity prediction),
  weighted by a Bayesian heteroscedastic loss whose per-band log-variances are predicted
  by Morpheus from the statistical signature of the current noisy state.
- **Backbone:** HDiT with neighbourhood attention on axial wavelet slices and
  spatio-depth factorised attention across slices.
- **Conditioning:** subject age (numeric, normalised to the training range).
- **Sampling:** Heun (2nd order) or Euler ODE integration, with optional
  uncertainty-minimisation guidance from Morpheus.

The release is a one-factor architecture ablation over a shared baseline. All
variants use the same CFM objective, Morpheus scheduler and HDiT backbone; each
changes a single axis.


| Checkpoint | Variant | Changes vs. baseline | Params | Full-res inference VRAM¹ | Status |
|---|---|---|---|---|---|
| `WaveDiT-Base.pth` | baseline | patch 8×8, depth 2/2, width 1024 | 142M | ~3.1 GB (runs from 4 GB) | ✅ trained |
| `WaveDiT-FinePatch.pth` | finer patches | patch 4×4 (4× tokens) | 142M | ~8.4 GB (runs from 10 GB) | ✅ trained |
| `WaveDiT-FinePatch2.pth` | finest patches (warm-started) | patch 2×2 (16× tokens) | 142M | ~27 GB (runs from 32 GB) | 🟡 pre-release · ⏳ training |
| `WaveDiT-Deep.pth` | deeper | depth 4/4 | 262M | ~3.1 GB (runs from 4 GB) | ✅ trained |
| `WaveDiT-Wide.pth` | wider | width 2048, d_ff 8192 | 506M | ~5.6 GB (runs from 8 GB) | ✅ trained |



¹ Peak VRAM for full-resolution (224³) generation, batch 1, bf16, 10-step Heun
(`torch.cuda.max_memory_reserved`). The HDiT backbone is **highly scalable**: because
patch size, width and depth are config knobs over a compact wavelet representation, WaveDiT fits
a wide range of hardware budgets: **full-resolution inference runs on GPUs from 4 GB
upward** (Base), and the same configs scale training down to modest GPUs by adjusting
batch size / variant. No high-end accelerator is required to *use* the models.


### FinePatch2: warm-started, not trained from scratch

`WaveDiT-FinePatch2` takes the patch axis to its finest setting (2×2 patches, a 56×56
token grid, 16× the tokens of `Base`). It was **not** trained from scratch: it was
**warm-started by weight inheritance** from `WaveDiT-FinePatch` (4×4). The entire HDiT
transformer body transfers 1:1, and only the two patch projections are resized to the
finer grid with a FlexiViT pseudo-inverse resize, so optimisation resumes already in
distribution instead of from noise. In practice this **cut wall-clock training time
drastically** versus a from-scratch run, while the finer token grid produces
**very high quality samples**. The procedure is `scripts/weight_inheritance.py` in the
[GitHub repository](https://github.com/sisinflab/WaveDiT).


## How to use

The checkpoint is self-contained (architecture + condition metadata embedded), and the
generation code lives in the [GitHub repository](https://github.com/sisinflab/WaveDiT):

```bash
git clone https://github.com/sisinflab/WaveDiT && cd WaveDiT
pip install -r requirements.txt
```

```python
from huggingface_hub import hf_hub_download

# pick a variant: WaveDiT-Base | WaveDiT-FinePatch | WaveDiT-FinePatch2 | WaveDiT-Deep | WaveDiT-Wide
ckpt = hf_hub_download("danesed/WaveDiT", "WaveDiT-Base.pth", revision="main")
```

```bash
# 4 volumes at age 45, cropped to the standard 182x218x182 MNI grid.
# NOTE: global flags (--num-flow-steps, --sampler, --save-size, ...) go BEFORE the subcommand.
PYTHONPATH=. python scripts/generate.py "$CKPT" out/ \
    --num-flow-steps 10 --sampler heun --save-size 182 218 182 \
    specific --conditions "age=45.0" --num-samples 4

# Linear age sweep, one volume per step
PYTHONPATH=. python scripts/generate.py "$CKPT" out/ \
    linear --condition age --min 6 --max 95 --num 100
```

No NATTEN? Set `WAVEDIT_NA_BACKEND=torch` to use the built-in pure-PyTorch neighbourhood
attention (e.g. on Spaces); the same checkpoint produces equivalent volumes.

Volumes are written as NIfTI (`.nii.gz`) with intensities in `[0, 1]`.
The checkpoint loads with the `torch.load` default `weights_only=True` (PyTorch ≥ 2.6).

## Samples

Age-conditioned synthesis with `WaveDiT-FinePatch` at a fixed seed;
rows are axial · coronal · sagittal mid-slices, columns span ages 6→95.

![WaveDiT-FinePatch aging](assets/samples/WaveDiT-FinePatch_aging.png)

## Training data

Trained on cognitively normal T1-weighted scans pooled from **OASIS-3**, **ADNI** and
**OpenBHB** (ages 6–95). These datasets are governed by data-use agreements and are
**not redistributed** here or in the GitHub repository; access must be requested from the
original providers.

## Intended use and limitations

- **Research use only.** This model is intended for research on generative modelling and
  data augmentation in neuroimaging. It is **not a medical device** and must not be used
  for diagnosis, treatment planning or any clinical decision-making.
- Synthetic volumes reflect the demographic and acquisition characteristics of the
  training cohorts (healthy/cognitively normal subjects, specific scanners and
  protocols); they may not generalise to other populations, pathologies or modalities.
- Age conditioning interpolates within the training age range; values outside it are
  clamped.

## Citation

```bibtex
@misc{danese2026waveditdistributionawarewaveletflow,
      title={WaveDiT: Distribution-Aware Wavelet Flow Matching for Efficient 3D Brain MRI Synthesis},
      author={Danilo Danese and Angela Lombardi and Giuseppe Fasano and Matteo Attimonelli and Tommaso Di Noia},
      year={2026},
      eprint={2606.08670},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.08670},
}
```
