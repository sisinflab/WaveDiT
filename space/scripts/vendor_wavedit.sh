#!/usr/bin/env bash
# Vendor the wavedit/ package into this Space and apply the lazy-wandb patch to
# BOTH files that import wandb at module top level.
#
# Run from the Space root (the dir that contains app.py):
#     bash scripts/vendor_wavedit.sh /path/to/WaveDiT/repo
#
# Default source repo: the parent of this Space dir's parent (../..).
#
# WHY TWO FILES:
#   - wavedit/models/wavelet_flow_matching.py imports wandb (used in training loss
#     logging only; the sampling path never touches it).
#   - wavedit/training/trainer.py ALSO imports wandb at module top. This matters
#     because wavedit/training/__init__.py does `from .trainer import Trainer`, so
#     `from wavedit.training.checkpoint import load_model_weights` (used by app.py)
#     transitively imports trainer.py. Without patching trainer.py the Space would
#     need wandb installed just to import the weight loader.
#
# WHY WE COPY THE WHOLE TREE (no per-file exclusions):
#   wavedit/models/hdit/transformer.py does `from . import flags, flops`, so
#   flops.py IS on the generation import path (it is pure stdlib; harmless).
#   Copying everything avoids accidentally dropping an import-path module.
set -euo pipefail

SPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_REPO="${1:-$(cd "$SPACE_ROOT/.." && pwd)}"

SRC_PKG="$SRC_REPO/wavedit"
DST_PKG="$SPACE_ROOT/wavedit"

if [[ ! -d "$SRC_PKG" ]]; then
  echo "ERROR: source package not found at $SRC_PKG" >&2
  echo "Pass the WaveDiT repo path: bash scripts/vendor_wavedit.sh /path/to/WaveDiT" >&2
  exit 1
fi

echo "Vendoring $SRC_PKG -> $DST_PKG"
rm -rf "$DST_PKG"
# Copy the whole package (drop caches only) so no import-path module is accidentally
# dropped, then prune the two subtrees that are NEVER imported on the sampling path.
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SRC_PKG/" "$DST_PKG/"

# Prune training-only / dependency-heavy subtrees that the sampling path never imports:
#   - wavedit/evaluation/ -> visualization.py has a bare `import wandb` (a latent
#     landmine: wandb is intentionally absent from the Space requirements). Nothing on
#     the gen path imports it (generator.py, models/__init__, training/__init__ and
#     trainer.py do NOT import evaluation), so deleting it is the cleanest fix.
#   - wavedit/data/ -> dataset/transform/collate code; never imported at sample time.
# (Verified: trainer.py only *mentions* evaluation in a comment; it does not import it.)
rm -rf "$DST_PKG/evaluation" "$DST_PKG/data"

# --- Apply the lazy-wandb patch (belt-and-braces; evaluation/ is normally pruned) ---
python3 - "$DST_PKG" <<'PY'
import re
import sys
from pathlib import Path

dst = Path(sys.argv[1])

LAZY = (
    "try:  # wandb is training-only; the sampling path never touches it.\n"
    "    import wandb\n"
    "except ImportError:  # ZeroGPU Space ships without wandb (vendored, lazy-import patch).\n"
    "    wandb = None"
)

def patch(rel: str) -> None:
    p = dst / rel
    if not p.exists():
        print(f"  WARN: {rel} not found, skipping")
        return
    s = p.read_text(encoding="utf-8")
    # 1) top-level `import wandb` (exact line) -> lazy try/except.
    new, n = re.subn(r"(?m)^import wandb[ \t]*$", LAZY, s, count=1)
    if n == 0 and "import wandb" not in s:
        print(f"  {rel}: no top-level `import wandb` (already patched?)")
    s = new
    # 2) guard every bare `wandb.log(...)` call so it no-ops when wandb is None.
    #    Calls already behind `if self.use_wandb:` stay correct; this is belt-and-braces.
    def _guard(m: re.Match) -> str:
        indent, call = m.group(1), m.group(2)
        return f"{indent}if wandb is not None:\n{indent}    {call}"
    s = re.sub(r"(?m)^([ \t]*)(wandb\.log\([^\n]*\))[ \t]*$", _guard, s)
    p.write_text(s, encoding="utf-8")
    print(f"  patched {rel}")

patch("models/wavelet_flow_matching.py")
patch("training/trainer.py")
# evaluation/visualization.py is normally pruned above; patch it too IF it survived
# (e.g. someone removed the prune), so the tree never carries a bare `import wandb`.
patch("evaluation/visualization.py")
PY

# --- Sanity: the two gen-path files must be import-safe without wandb ---
for f in "models/wavelet_flow_matching.py" "training/trainer.py"; do
  grep -q "wandb = None" "$DST_PKG/$f" || { echo "ERROR: lazy wandb patch missing in $f" >&2; exit 1; }
done
# NO bare top-level `import wandb` may survive ANYWHERE in the vendored tree (wandb is
# absent from the Space requirements; any unpatched import is a latent import crash).
if grep -rn "^import wandb" "$DST_PKG" >/dev/null 2>&1; then
  echo "ERROR: a bare top-level 'import wandb' survives in the vendored tree:" >&2
  grep -rn "^import wandb" "$DST_PKG" >&2
  exit 1
fi
# flops.py is on the gen path (imported by transformer.py); it must be present.
[[ -f "$DST_PKG/models/hdit/flops.py" ]] || { echo "ERROR: models/hdit/flops.py missing (it is on the import path)" >&2; exit 1; }

# Byte-compile the gen-path modules as a final guard.
python3 -m py_compile \
  "$DST_PKG/models/wavelet_flow_matching.py" \
  "$DST_PKG/training/trainer.py" \
  "$DST_PKG/training/checkpoint.py" \
  "$DST_PKG/models/factory.py" \
  "$DST_PKG/generation/generator.py" \
  "$DST_PKG/models/hdit/transformer.py" \
  "$DST_PKG/models/hdit/flops.py"

echo "Done. Vendored package is at $DST_PKG (lazy-wandb patch applied; evaluation/ + data/ pruned)."
echo "Reminder: do NOT copy pyproject.toml / setup.py (they pin torch==2.6.0)."
