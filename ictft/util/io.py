from typing import Tuple
import os
import torch
import numpy as np


def parse_obj(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parse an obj file
    """
    def parse_poly(line: str):
        """
        Parse a polygon, also triangulate
        """
        poly_info = line.split(' ')[1:]
        vp_idx = []
        vt_idx = []
        if len(poly_info) == 3:
            for info in poly_info:
                vp = info.split('/')[0]
                # obj counts from 1
                vp_idx += [int(vp) - 1]
                vt = info.split('/')[1]
                # obj counts from 1
                vt_idx += [int(vt) - 1]
            vp_idx = [torch.tensor(vp_idx)]
            vt_idx = [torch.tensor(vt_idx)]
        elif len(poly_info) == 4:
            v1, v2, v3, v4 = poly_info
            # obj counts from 1
            vp_idx += [torch.tensor([int(v1.split('/')[0]) - 1, int(v2.split('/')[0]) - 1, int(v3.split('/')[0]) - 1])]
            vp_idx += [torch.tensor([int(v3.split('/')[0]) - 1, int(v4.split('/')[0]) - 1, int(v1.split('/')[0]) - 1])]
            vt_idx += [torch.tensor([int(v1.split('/')[1]) - 1, int(v2.split('/')[1]) - 1, int(v3.split('/')[1]) - 1])]
            vt_idx += [torch.tensor([int(v3.split('/')[1]) - 1, int(v4.split('/')[1]) - 1, int(v1.split('/')[1]) - 1])]
        return vp_idx, vt_idx

    if not os.path.exists(filepath):
        print(f'Error: The specified filepath is invalid: {filepath}')
    
    with open(filepath, 'r') as infd:
        obj_lines = infd.readlines()
    v_pos = []
    v_uvs = []
    shapes = { }
    for obj_line in obj_lines:
        obj_line = obj_line.replace('\n', '')
        if obj_line.startswith('v') and not (obj_line.startswith('vt') or obj_line.startswith('vn')):
            vpos = obj_line.split(' ')[1:]
            vpos = [float(v) for v in vpos]
            assert len(vpos) == 3
            v_pos += [torch.tensor([vpos])]
        elif obj_line.startswith('vt'):
            vt = obj_line.split(' ')[1:]
            vt = [float(v) for v in vt]
            v_uvs += [torch.tensor([vt])]
        elif obj_line.startswith('usemtl'):
            # here we have a new shape
            curr_shapename = obj_line.split(' ')[-1]
            shapes[curr_shapename] = {'t_pos_idx': [], 't_uvs_idx': []}
        elif obj_line.startswith('f'):
            vp_idx, vt_idx = parse_poly(obj_line)
            for vp_i in vp_idx:
                shapes[curr_shapename]['t_pos_idx'] += [vp_i]
            for vt_i in vt_idx:
                shapes[curr_shapename]['t_uvs_idx'] += [vt_i]
        else:
            continue
    v_pos = torch.vstack(v_pos).float().cuda()
    v_uvs = torch.vstack(v_uvs).float().cuda()
    for name in shapes:
        shapes[name]['t_pos_idx'] = torch.vstack(shapes[name]['t_pos_idx']).long().cuda()
        shapes[name]['t_uvs_idx'] = torch.vstack(shapes[name]['t_uvs_idx']).long().cuda()
    return v_pos, v_uvs, shapes
        
    
def load_model(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load ict facekit's generic model

    :param dir The directory with ICTFaceKit's data.
    :return
    """
    v_pos, v_uvs, shapes = parse_obj(filepath)
    v_uvs[:, 1] = 1.0 - v_uvs[:, 1]
    return v_pos, v_uvs, shapes


