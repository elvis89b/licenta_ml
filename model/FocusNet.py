"""Protocol-specific FocusNet model registry.

Center-wise experiments use FocusNetEFPM.
Modality-wise experiments use FocusNetUGEL.
The training and testing scripts import their model explicitly.
"""

from .FocusNet_EFPM import FocusNet as FocusNetEFPM
from .FocusNet_UGEL import FocusNet as FocusNetUGEL

__all__ = ["FocusNetEFPM", "FocusNetUGEL", "build_focusnet"]


def build_focusnet(protocol: str):
    protocol = protocol.strip().lower().replace("-", "_")

    if protocol in {"center", "center_wise", "efpm"}:
        return FocusNetEFPM()

    if protocol in {"modality", "modality_wise", "ugel"}:
        return FocusNetUGEL()

    raise ValueError(
        f"Unknown protocol: {protocol!r}. "
        "Use 'center_wise'/'efpm' or 'modality_wise'/'ugel'."
    )
