"""Torch-free tests for the YAML config layer. Run from the repo root:

    python tests/test_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wavedit import Config
from wavedit.config import FLOW_FORMULATIONS


def test_shipped_configs_load():
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    yamls = sorted(config_dir.glob("*.yaml"))
    assert yamls, "no configs found"
    for path in yamls:
        cfg = Config.from_yaml(path)
        assert cfg.model.flow in FLOW_FORMULATIONS
        assert len(cfg.data.image_size) == 3 and isinstance(cfg.data.image_size, tuple)
        assert len(cfg.model.patch_size) == 2
        # Latent slices must be divisible by the patch size (mirrors the model factory check).
        d, h, w = (s // 2 for s in cfg.data.image_size)
        assert h % cfg.model.patch_size[0] == 0 and w % cfg.model.patch_size[1] == 0
        assert cfg.model.levels and cfg.model.mapping
        print(f"  OK {path.name}: flow={cfg.model.flow} image={cfg.data.image_size}")


def test_roundtrip_and_coercion():
    cfg = Config.from_dict({
        "run_name": "rt",
        "data": {"conditions": {"Age": "Numeric"}, "image_size": [64, 64, 64]},
        "model": {"patch_size": [8, 8], "flow": "cfm", "levels": [{"depth": 1, "width": 8}], "mapping": {"depth": 1, "width": 8}},
    })
    assert cfg.data.conditions == {"age": "numeric"}          # lowercased
    assert cfg.data.image_size == (64, 64, 64)                # list -> tuple
    again = Config.from_dict(cfg.to_dict())                   # round-trips through a checkpoint dict
    assert again.to_dict() == cfg.to_dict()
    print("  OK round-trip + coercion")


def test_validation_rejects_bad_values():
    for bad in [{"precision": "int8"}, {"model": {"flow": "nope"}}, {"sampling": {"sampler": "rk4"}},
                {"data": {"image_size": [1, 2]}}, {"data": {"conditions": {"x": "weird"}}}]:
        try:
            Config.from_dict(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected validation error for {bad}")
    print("  OK validation rejects bad values")


if __name__ == "__main__":
    test_shipped_configs_load()
    test_roundtrip_and_coercion()
    test_validation_rejects_bad_values()
    print("config tests passed")
