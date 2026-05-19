from typing import Protocol, Callable, Dict, Any
from dataclasses import dataclass

import torch
import t3d
import mlf

import os

@dataclass
class TransferState:
    flame_model: t3d.mms.Flame2023 = None
    ict_model: t3d.mms.ICTModel = None
    output_dir: str = ""
    device: torch.device=torch.device('cuda')

    lr: float = None
    w_data: float = None
    w_edge: float = None
    w_laplacian: float = None
    w_normal: float = None
    knn_k: int=1
    w_lmks: float = None
    w_arap: float = None
    nrigid_iters: int = None
    update_correspondencies_every: int=None


def state_init(
    flame2023_kwargs: Dict[str, Any],
    ict_model_kwargs: Dict[str, Any],
    output_dir: str,
    device: list[int],
    nrigid_iters: int=5000,
    lr: float=1.0e-3,
    w_data: float=1.0,
    w_edge: float=1.0,
    w_laplacian: float=0.1,
    w_normal: float=0.01,
    w_lmks: float=100.0,
    w_arap: float=10.0,
    knn_k: int=1,
    update_correspondences_every: int=10
) -> TransferState:
    """
    Initialize a TransferState object
    """
    state = TransferState()
    mlf.log.info("Configuring FLAME2023 model...")
    state.flame_model = t3d.mms.Flame2023(**flame2023_kwargs)
    mlf.log.info("FLAME configured.")
    mlf.log.info("Configuring ICT FaceKit model...")
    state.ict_model = t3d.mms.ICTModel(**ict_model_kwargs)
    mlf.log.info("ICT FaceKit configured.")
    
    state.output_dir = output_dir
    if not os.path.exists(state.output_dir):
        os.mkdir(state.output_dir)

    state.device = mlf.util.conf.get_device(device)
    state.flame_model = state.flame_model.to(state.device)
    state.ict_model = state.ict_model.to(state.device)

    state.lr = lr
    state.w_data = w_data
    state.w_edge = w_edge
    state.w_laplacian = w_laplacian
    state.w_normal = w_normal
    state.w_lmks = w_lmks
    state.w_arap = w_arap
    state.knn_k = knn_k
    state.update_correspondencies_every = update_correspondences_every
    state.nrigid_iters = nrigid_iters
    return state


