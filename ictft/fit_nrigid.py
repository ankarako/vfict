from typing import Tuple
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)
from pytorch3d.ops import sample_points_from_meshes, knn_points

from ictft.state import TransferState
import mlf
from tqdm import tqdm


def compute_vertex_neighbors(faces: torch.Tensor, nverts: int) -> torch.Tensor:
    """
    """
    neighbors = [set() for _ in range(nverts)]

    for face in faces:
        v0, v1, v2 = face.tolist()
        neighbors[v0].update([v1, v2])
        neighbors[v1].update([v0, v2])
        neighbors[v2].update([v0, v1])
    
    neighbors = [torch.tensor(list(n), dtype=torch.long, device=faces.device)
                 if len(n) > 0 else torch.tensor([], dtype=torch.long, device=faces.device)
                 for n in neighbors]

    return neighbors

def compute_arap_loss(source_verts: torch.Tensor,
                      deformed_verts: torch.Tensor,
                      neighbors: list) -> torch.Tensor:
    """
    Compute ARAP (As-Rigid-As-Possible) loss
    Penalizes non-rigid deformations by measuring deviation from local rigidity

    Args:
        source_verts: (V, 3) original vertices
        deformed_verts: (V, 3) deformed vertices
        neighbors: List of neighbor indices for each vertex

    Returns:
        arap_loss: Scalar loss value
    """
    loss = 0.0
    count = 0

    for i, neighs in enumerate(neighbors):
        if len(neighs) == 0:
            continue

        # Original and deformed edges from vertex i to its neighbors
        source_edges = source_verts[neighs] - source_verts[i]  # (N, 3)
        deformed_edges = deformed_verts[neighs] - deformed_verts[i]  # (N, 3)

        # Compute optimal rotation via Procrustes (SVD)
        H = source_edges.T @ deformed_edges  # (3, 3)
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if torch.det(R) < 0:
            Vt = Vt.clone()
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Rotated source edges should match deformed edges
        rotated_edges = source_edges @ R

        # Measure deviation from rigidity
        loss += ((deformed_edges - rotated_edges) ** 2).sum()
        count += len(neighs)

    return loss / (count + 1e-10)


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
    device = state.device
    deformed = state.flame_model.v_template.clone().requires_grad_(True)
    verts_src = state.flame_model.v_template.clone()
    neighbors = compute_vertex_neighbors(state.flame_model.faces, verts_src.shape[0])

    optim = torch.optim.Adam([deformed], lr=state.lr)

    verts_tgt_batch = state.ict_model.v_pos.unsqueeze(0)
    correspondences = None

    mlf.log.info(f"Starting non-rigid ICP optim ({state.nrigid_iters} iterations)")
    optim_loop = tqdm(range(state.nrigid_iters), total=state.nrigid_iters, desc="Non-Rigid Optim")
    for iteration in optim_loop:
        optim.zero_grad()

        if iteration % state.update_correspondencies_every == 0:
            with torch.no_grad():
                # find nearest neighbors
                deformed_batch = deformed.unsqueeze(0)
                knn_res = knn_points(deformed_batch, verts_tgt_batch, K=state.knn_k)

                # get indices of nearest neighbors
                knn_idx = knn_res.idx[0]

                # get corresponding target vertices
                if state.knn_k == 1:
                    correspondences = state.ict_model.v_pos[knn_idx.squeeze(-1)]
                else:
                    correspondences = state.ict_model.v_pos[knn_idx].mean(dim=1)
        
        mesh_src = Meshes([deformed], [state.flame_model.faces])

        loss_data = torch.nn.functional.mse_loss(deformed, correspondences)
        loss_edge = mesh_edge_loss(mesh_src)
        loss_lap = mesh_laplacian_smoothing(mesh_src, method='uniform')
        loss_nrm = mesh_normal_consistency(mesh_src)
        loss_arap = compute_arap_loss(verts_src, deformed, neighbors)
        loss_lmk = torch.tensor(0.0, device=device)

        loss = (
            state.w_data * loss_data +
            state.w_edge * loss_edge +
            state.w_laplacian * loss_lap +
            state.w_normal * loss_nrm +
            state.w_arap * loss_arap
        )

        if iteration % 10 == 0:
            optim_loop.set_postfix({'loss': f"{loss.detach().item():.4f}"})
        # TODO: lmks loss
        loss.backward()
        optim.step()

    return deformed.detach()

    

