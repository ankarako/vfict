from typing import Tuple
import torch
from pytorch3d.structures import Meshes

from ictft.state import TransferState

def point_to_face(
    points: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance from points to triangles and barycentrics

    Args:
        points: (P, 3) query points
        v0, v1, v2: (F, 3) triangle vertices
    
    Returns:
        distances: (P, F) distances to each triangle
        bary_coords: (P, F, 3) barycentric coordinates
    """
    # reshape for broadcasting
    p = points.unsqueeze(1)
    v0 = v0.unsqueeze(0)
    v1 = v1.unsqueeze(0)
    v2 = v2.unsqueeze(0)

    # triangle edges
    e0 = v1 - v0
    e1 = v2 - v0

    # vector from v0 to point
    v0p = p - v0

    # compute dot products
    a = (e0 * e0).sum(-1)  # (1, F)
    b = (e0 * e1).sum(-1)
    c = (e1 * e1).sum(-1)
    d = (e0 * v0p).sum(-1)  # (P, F)
    e = (e1 * v0p).sum(-1)

    # solve for barycentrics
    det = a * c - b * b
    u = (c * d - b * e) / (det + 1e-10)
    v = (a * e - b * d) / (det + 1e-10)
    w = 1.0 - u - v

    # clamp to valid barycentrics
    u = torch.clamp(u, 0, 1)
    v = torch.clamp(v, 0, 1)

    # ensure u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1
    u[mask] = u[mask] / sum_uv[mask]
    v[mask] = v[mask] / sum_uv[mask]
    w = 1.0 - u - v

    # stack barys
    bary_coords = torch.stack([w, u, v], dim=-1)
    
    # compute closest point to triangle
    closest = w.unsqueeze(-1) * v0 + u.unsqueeze(-1) * v1 + v.unsqueeze(-1) * v2
    distances = torch.norm(p - closest, dim=-1)
    return distances, bary_coords


def project_to_mesh(
    query_points: torch.Tensor,
    target_verts: torch.Tensor,
    target_faces: torch.Tensor,
    batch_size: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project query points onto the target mesh surface

    Args:
        query_points: (Q, 3) the points to project
        target_verts: (V, 3) the target mesh vertices.
        target_faces: (F, 3) the target mesh face indices.
        batch_size: Number of query points to process at once

    Returns:
        closest_face_idx: (Q) index of the closest face for each query point
        bary_coords: (Q, 3) barycentric coordinates in closest surface
        distances: (Q) distance to closest face
    """
    num_queries = query_points.shape[0]
    device = query_points.device

    # get triangle verts
    v0 = target_verts[target_faces[:, 0]]
    v1 = target_verts[target_faces[:, 1]]
    v2 = target_verts[target_faces[:, 2]]

    closest_face_idx = torch.zeros([num_queries], dtype=torch.long, device=device)
    bary_coords = torch.zeros([num_queries, 3], device=device)
    min_distances = torch.full([num_queries], float('inf'), device=device)

    # process in batches
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_points = query_points[start_idx:end_idx]

        # compute distances and barys
        dists, barys = point_to_face(batch_points, v0, v1, v2)

        # find closest face for each input
        batch_min_dists, batch_face_idx = dists.min(dim=1)

        # store results
        min_distances[start_idx:end_idx] = batch_min_dists
        closest_face_idx[start_idx:end_idx] = batch_face_idx

        # get barys for closest faces
        batch_idx = torch.arange(batch_points.shape[0], device=device)
        bary_coords[start_idx:end_idx] = barys[batch_idx, batch_face_idx]
    return closest_face_idx, bary_coords, min_distances


def non_rigid_icp(
    state: TransferState,
    num_iters: int=10,
    stiffness: float = 1.0
) -> torch.Tensor:
    """
    Perform non-rigid ICP from source (FLAME) to target (ICT)

    Returns:
        The deformed FLAME vertices
    """
    deformed = state.flame_model.v_template.clone()
    for iteration in range(num_iters):
        closest_face_idx, bary_coords, distances = project_to_mesh(
            deformed, state.ict_model.v_pos, state.ict_model.t_pos_idx
        )

        # compute corresponding points on target
        faces = state.ict_model.t_pos_idx[closest_face_idx]
        v0 = state.ict_model.v_pos[faces[:, 0]]
        v1 = state.ict_model.v_pos[faces[:, 1]]
        v2 = state.ict_model.v_pos[faces[:, 2]]

        correspondences = (
            bary_coords[:, 0:1] * v0 +
            bary_coords[:, 1:2] * v1 + 
            bary_coords[:, 2:3] * v2
        )

        # simple deformation: weighted average
        alpha = 0.5
        deformed = (1 - alpha) * deformed + alpha * correspondences

        if iteration % 5 == 0:
            mean_dist = distances.mean().item()
            print(f"  Iteration {iteration}: mean distance = {mean_dist:.6f}")
    return deformed

    

