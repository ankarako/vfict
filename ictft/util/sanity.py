# Tools for sanity checks on the uv space
# as ICT's uv coordinates are a little bit strange
from typing import Dict, Any
import torch

def fix_shape_uvs(shape_data: Dict[str, Any], v_uvs) -> torch.Tensor:
    """
    """
    # gather all the uvs that correspond to this shape
    uvs = v_uvs[shape_data['t_uvs_idx'], ...]
    max = uvs.max()
    min = uvs.min()
    return uvs