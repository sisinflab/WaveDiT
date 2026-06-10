# Hugging Face release checklist

Steps to publish WaveDiT on the Hub under the `danesed` namespace, once the
config sweep has selected the release checkpoint(s).

## 0. Prerequisites

```bash
pip install -U huggingface_hub   # ships the `hf` CLI (huggingface-cli is deprecated)
hf auth login
```

## Naming

Configs, run names and checkpoint folders all use the descriptive variant names
(`configs/cfm_<Variant>.yaml` -> run_name `WaveDiT_CFM_<Variant>`):

| Checkpoint folder | Release name |
|---|---|
| `checkpoints/WaveDiT_CFM_Base/best.pth` | `WaveDiT-Base.pth` |
| `checkpoints/WaveDiT_CFM_FinePatch/best.pth` | `WaveDiT-FinePatch.pth` |
| `checkpoints/WaveDiT_CFM_Deep/best.pth` | `WaveDiT-Deep.pth` |
| `checkpoints/WaveDiT_CFM_Wide/best.pth` | `WaveDiT-Wide.pth` |

## 1. Slim the chosen checkpoint(s)

Strips optimiser/scheduler state; keeps weights + the self-contained metadata
(`config`, `condition_config`, `condition_ranges`, `categorical_maps`, `null_conditions`):

```bash
slim() {  # slim() <run_folder> <release_name>
  PYTHONPATH=. python tools/slim_checkpoint.py "checkpoints/$1/best.pth" "$2.pth"
  PYTHONPATH=. python scripts/generate.py "$2.pth" /tmp/hf_check/"$2" \
      --save-size 182 218 182 \
      specific --conditions "age=45.0" --num-samples 1  # sanity-check
}
slim WaveDiT_CFM_Base WaveDiT-Base
slim WaveDiT_CFM_FinePatch WaveDiT-FinePatch
slim WaveDiT_CFM_Deep WaveDiT-Deep
slim WaveDiT_CFM_Wide WaveDiT-Wide
```

(Publish all four as an ablation, or just the winner — decide after the sweep.)

## 2. Create the model repo and upload (incremental is fine)

All variants live in one repo (`danesed/WaveDiT`) with the model card as README.
A model repo is a git repo: weights can be added in **separate commits as each run
finishes**, so you can release `WaveDiT-Base` first (it trains fastest) and append
the others later — already-downloaded files are unaffected.

```bash
hf repos create danesed/WaveDiT --repo-type model
hf upload danesed/WaveDiT hf/MODEL_CARD.md README.md   # card first
hf upload danesed/WaveDiT assets/WaveDiT_Architecture.png assets/WaveDiT_Architecture.png  # optional figure

# First release: just Base
hf upload danesed/WaveDiT WaveDiT-Base.pth WaveDiT-Base.pth

# Later, as each run finishes — one commit each:
hf upload danesed/WaveDiT WaveDiT-Deep.pth WaveDiT-Deep.pth
hf upload danesed/WaveDiT WaveDiT-FinePatch.pth WaveDiT-FinePatch.pth
hf upload danesed/WaveDiT WaveDiT-Wide.pth WaveDiT-Wide.pth
```

Each time you add a variant, update the **Status** column in the model card table and
re-upload it, so the card never lists a file that isn't there yet.

## 3. Tag the release (reproducible `revision=` pinning)

During the incremental phase, early users download with `revision="main"`. Only once
**all** the weights you intend to ship are uploaded, cut a single immutable `v1.0` tag
— at that point the card's `revision="v1.0"` is correct and frozen. Never move a tag
that's already published (it breaks reproducibility for anyone who pinned it).

```python
from huggingface_hub import HfApi
HfApi().create_tag("danesed/WaveDiT", tag="v1.0", repo_type="model")
```

## 4. Paper page

1. Visit https://huggingface.co/papers/2606.08670 — if not indexed yet, this indexes it
   (or submit via https://huggingface.co/papers/submit).
2. Click your author name → **claim authorship** → confirm in settings
   (https://huggingface.co/settings/papers); validation is manual on HF's side.
3. The model is linked to the paper page automatically: the model card README contains
   the arXiv URL, which the Hub parses into an `arxiv:2606.08670` tag. Add the GitHub URL
   and project page on the paper page once claimed.

## 5. (Optional) Spaces demo

ZeroGPU requires a PRO subscription or an HF-side grant — reply to Niels' offer for the
grant. Constraints: Gradio SDK only, GPU work inside functions decorated with
`@spaces.GPU(duration=...)` (default budget 60 s; full-volume Heun sampling will need
more), model moved to `cuda` at module import. The demo app lives in the Space repo, not
in this repository.

## Notes

- The checkpoint loads under the torch ≥ 2.6 `torch.load` default `weights_only=True`
  (verified: metadata is plain Python types + tensors). Do not add custom classes to the
  checkpoint payload.
- Do NOT upload anything from `data/` (OASIS-3 / ADNI / OpenBHB are non-redistributable)
  or `wandb/`.
