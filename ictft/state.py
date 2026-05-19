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


def state_init(
    flame2023_kwargs: Dict[str, Any],
    ict_model_kwargs: Dict[str, Any],
    output_dir: str,
    device: list[int]
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
    return state


