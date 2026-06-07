"""WaveDiT: Distribution-Aware Wavelet Flow Matching for 3D Brain MRI Synthesis.

The top-level package intentionally only exposes the lightweight, torch-free
:class:`~wavedit.config.Config`. Heavyweight components (models, training,
generation) are imported from their submodules where needed, so that simply
``import wavedit`` does not require a CUDA/torch stack.
"""

from .config import Config

__version__ = "1.0.0"
__all__ = ["Config", "__version__"]
