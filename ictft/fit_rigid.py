from typing import Tuple
import torch
from ictft.state import TransferState
import t3d

def rigid_alignment_lmk(state: TransferState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the rigid alignment (rotation + translation + scale) from FLAME
    to ICT using procrustes, based on the models' multi-pie landmarks

    Args:
        state: A TransferState object holding the FLAME and ICT models

    Returns:
        The rotation, translationm and scale for tranforming FLAME to ICT.
    """
    lmks_ict = t3d.mms.ICTLandmarks.get_ict_lmks(state.ict_model.v_pos)
    # perform a forward pass on FLAME on the template
    v_pos_fl, lmks_flame = state.flame_model.forward(
        id_params=torch.zeros([1, state.flame_model.nid_params], device=state.device),
        ex_params=torch.zeros([1, state.flame_model.nex_params], device=state.device),
        rotation=torch.zeros([1, 3], device=state.device),
        neck=torch.zeros([1, 3], device=state.device),
        jaw=torch.zeros([1, 3], device=state.device),
        eyes=torch.zeros([1, 6], device=state.device),
        translation=torch.zeros([1, 3], device=state.device),
        return_lmks=True
    )
    lmks_flame = lmks_flame.squeeze()[:-2, :]

    # source is FLAME, target is ICT
    center_src = lmks_flame.mean(dim=0)
    center_tgt = lmks_ict.mean(dim=0)
    centered_src = lmks_flame - center_src
    centered_tgt = lmks_ict - center_tgt

    # compute scale
    scale_src = torch.norm(centered_src)
    scale_tgt = torch.norm(centered_tgt)
    scale = scale_tgt / scale_src

    # normalize for rotation computation
    normalized_src = centered_src / scale_src
    normalized_tgt = centered_tgt / scale_tgt

    # compute rotation via SVD
    H = normalized_src.T @ normalized_tgt
    U, S, Vt = torch.linalg.svd(H)
    rotation = Vt.T @ U.T

    # handle reflection
    if torch.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    # compute translation
    trans = center_tgt - scale * (rotation @ center_src)
    return rotation, trans, scale


def rigid_flame_to_ict(
    state: TransferState,
    rot: torch.Tensor,
    trans: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Rigid transform of FLAME with the specified
    rigid parameters.

    Returns:
        the transformed FLAME template vertices
    """
    v_pos_fl = state.flame_model.v_template
    aligned_v_pos_fl = scale * (v_pos_fl @ rot.T) + trans
    return aligned_v_pos_fl