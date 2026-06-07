"""HDiT (Hourglass Diffusion Transformer) backbone, adapted to 3D.

Vendored and extended from k-diffusion (https://github.com/crowsonkb/k-diffusion).
WaveDiT adds the ``SpatioTemporalAttentionSpec`` attention type and threads a depth
argument (``d_c``) through the transformer so slices of a 3D volume can attend
across the depth axis.
"""

from .layers import FourierFeatures
from .transformer import (
    GlobalAttentionSpec,
    ImageTransformerDenoiserModelV2,
    LevelSpec,
    MappingSpec,
    NeighborhoodAttentionSpec,
    ShiftedWindowAttentionSpec,
    SpatioTemporalAttentionSpec,
)

__all__ = [
    "FourierFeatures",
    "GlobalAttentionSpec",
    "ImageTransformerDenoiserModelV2",
    "LevelSpec",
    "MappingSpec",
    "NeighborhoodAttentionSpec",
    "ShiftedWindowAttentionSpec",
    "SpatioTemporalAttentionSpec",
]
