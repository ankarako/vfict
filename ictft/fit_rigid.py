import torch
from ictft.state import TransferState
import t3d

def rigid_alignment_lmk(state: TransferState) -> None:
    """
    Compute the rigid alignment (rotation + translation + scale) from FLAME
    to ICT using procrustes, based on the models' multi-pie landmarks

    Args:
        state: A TransferState object holding the FLAME and ICT models

    Returns:
        The rotation, translationm and scale for tranforming FLAME to ICT.
    """
    lmks_ict = t3d.mms.ICTLandmarks.get_ict_lmks(state.ict_model.template)
    # perform a forward pass on FLAME on the template
    