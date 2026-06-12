# WaveDiT Studio for macOS

WaveDiT Studio is a native macOS desktop app for generating and exploring
age-conditioned synthetic 3D brain MRI with [WaveDiT](https://arxiv.org/abs/2606.08670),
a wavelet-domain flow matching model (MICCAI 2026). Pick an age, generate a full
resolution volume on your Mac's GPU (Apple Silicon, PyTorch MPS), and explore it in an
interactive multiplanar and 3D viewer. Everything runs locally: after the one-time
weights download the app works fully offline.

> **Research use only.** WaveDiT Studio produces synthetic images for research and
> education. It is not a medical device, provides no diagnostic information, and must
> not be used for clinical decision making.

## Requirements

- Apple Silicon Mac (M1 or newer). Intel Macs are not supported.
- macOS 13 (Ventura) or newer.
- About 2 GB of free disk space for the app, plus the model weights you download
  (roughly 0.5 to 1.5 GB per model variant).
- 16 GB of RAM recommended for comfortable generation.

## Build

Build on the Mac that will run the app (or any Apple Silicon Mac). Only stock macOS
tools are used; Xcode and Homebrew are not required. A Python 3.11+ interpreter or
[uv](https://docs.astral.sh/uv/) must be available.

```bash
git clone https://github.com/danesed/WaveDiT.git
cd WaveDiT
git checkout macos-app
cd macos
./build.sh
```

The script creates a local build venv, verifies the vendored 3D viewer, renders the
app icon, freezes the app with PyInstaller, runs a self check, signs the bundle
(ad-hoc) and produces `dist/WaveDiT-Studio-1.0.0.dmg`.

## Install

1. Open the DMG and drag **WaveDiT Studio** into **Applications**.
2. First launch only: the app is ad-hoc signed (no Apple Developer certificate), so
   Gatekeeper shows a warning. Right-click (or control-click) the app in Applications,
   choose **Open**, then confirm. macOS remembers the choice.
   - Alternative, from a terminal:
     `xattr -dr com.apple.quarantine "/Applications/WaveDiT Studio.app"`

## First run

On first launch the app opens an onboarding panel and offers to download model weights
from [huggingface.co/danesed/WaveDiT](https://huggingface.co/danesed/WaveDiT) with live
progress. Start with **Base**; other variants (FinePatch, Deep, Wide) can be added or
removed at any time from the model picker. Weights are stored under
`~/Library/Application Support/WaveDiT Studio/checkpoints` and are downloaded only once.

## Usage tour

- **Age control**: the central control. Set the subject age and press **Generate**.
- **Model picker**: switch between downloaded WaveDiT variants; download or delete
  weights inline.
- **Advanced**: seed, number of flow steps, sampler (heun or euler), guidance scale and
  Morpheus uncertainty-guidance scale, for full control over the sampling trajectory.
- **Presets**: one-click sensible configurations (fast preview, balanced, max quality).
- **Aging time-lapse**: generate a sweep of volumes across an age range with a shared
  seed and scrub through the synthetic aging trajectory frame by frame.
- **Viewer**: multiplanar slices plus a 3D render with clip plane, colormap and gamma
  controls. Generation shows real per-step progress with a live time estimate.
- **Library**: every generation is kept on disk with its full settings. Reuse the exact
  settings of any item, re-open it in the viewer, or delete it.
- **Export**: save any volume as NIfTI (`.nii.gz`) through the native save dialog, ready
  for FSLeyes, ITK-SNAP, 3D Slicer or your own pipeline.

## Performance notes

- Per-step time depends on the model and your chip; the app calibrates itself and shows
  a live estimate before and during generation.
- Fewer flow steps mean faster results; the heun sampler costs roughly twice the model
  evaluations of euler per step but is more accurate at low step counts.
- **FinePatch is heavier** than the other variants and noticeably slower per step.
- The optional **bf16 performance mode** in Settings is experimental. If a generation
  fails in bf16 the app automatically retries in float32 and remembers the outcome.
- The very first generation after launching is slower while the model is loaded onto
  the GPU; later generations reuse the resident model.

## Dev mode (run from source, any OS)

The backend and UI run unbundled on macOS, Linux or Windows for development. From the
repository root:

```bash
pip install -r macos/requirements.txt
PYTHONPATH=macos python -m studio
```

Useful environment variables:

| Variable | Effect |
| --- | --- |
| `WAVEDIT_STUDIO_HEADLESS=1` | Server only, no window; open the printed URL in a browser. |
| `WAVEDIT_STUDIO_DEVICE` | Force `mps`, `cuda` or `cpu` (default: best available). |
| `WAVEDIT_STUDIO_CKPT_DIR` | Use an existing checkpoints directory instead of downloading. |
| `WAVEDIT_STUDIO_DATA_DIR` | Override the data directory (settings, library, exports). |
| `WAVEDIT_STUDIO_PORT` | Fixed server port (default: a random free port). |

`--selfcheck` prints versions and the selected device, then exits.

## Troubleshooting

- **"WaveDiT Studio cannot be opened"**: this is Gatekeeper reacting to the ad-hoc
  signature. Right-click the app, choose **Open**, confirm once. See Install above.
- **First generation is slow**: the model is loading onto the GPU. Subsequent
  generations are much faster.
- **Where is my data?** Everything lives in
  `~/Library/Application Support/WaveDiT Studio` (settings, checkpoints, library,
  exports). Volumes are plain `.nii.gz` files you can open with any NIfTI tool.
- **Reset the app**: quit, then delete that folder. The next launch starts fresh
  (weights will need to be downloaded again).
- **Free disk space**: delete unused model weights from the model picker, or remove
  old library items from the library shelf.

## Credits

WaveDiT Studio is a research artifact by SisInfLab, Politecnico di Bari, built on the
official WaveDiT implementation.

- Paper: [arXiv:2606.08670](https://arxiv.org/abs/2606.08670) /
  [Hugging Face paper page](https://huggingface.co/papers/2606.08670)
- Code: [github.com/danesed/WaveDiT](https://github.com/danesed/WaveDiT)
- Project page: [danesed.github.io/wavedit-page](https://danesed.github.io/wavedit-page/)
- Models: [huggingface.co/danesed/WaveDiT](https://huggingface.co/danesed/WaveDiT)
- Browser demo: [huggingface.co/spaces/danesed/WaveDiT-demo](https://huggingface.co/spaces/danesed/WaveDiT-demo)

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

The in-app 3D viewer is [Niivue](https://github.com/niivue/niivue). The HDiT backbone
is adapted from [k-diffusion](https://github.com/crowsonkb/k-diffusion).
