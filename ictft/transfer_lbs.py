import torch
from pytorch3d.ops import knn_points
from ictft.state import TransferState
from ictft.fit_nrigid import project_to_mesh


def transfer_neck_lbs(
    state: TransferState,
    flame_deformed_verts: torch.Tensor,
    falloff_range: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transfer FLAME's Neck joint to ICT and compute proper LBS weights with smooth falloff

    Args:
        state: TransferState with FLAME and ICT models
        flame_deformed_verts: (V_flame, 3) deformed FLAME vertices after non-rigid ICP
        falloff_range: Z-distance over which weights transition from 1.0 to 0.0

    Returns:
        ict_neck_weights: (V_ict,) Neck LBS weights for ICT vertices
        neck_joint: (3,) Neck joint position in ICT space
    """
    device = state.device

    # FLAME joint 1 is the Neck (joint 0 is root, joint 1 is neck)
    neck_joint_idx = 1

    # Get FLAME's neck weights (V_flame,)
    flame_neck_weights = state.flame_model.lbs_weights[:, neck_joint_idx]  # (V_flame,)

    # Get FLAME's neck joint regressor and compute joint position in deformed space
    # J_regressor is (num_joints, num_verts), sparse matrix
    j_regressor = state.flame_model.J_regressor  # (5, 5023)
    if hasattr(j_regressor, 'toarray'):
        j_regressor = torch.from_numpy(j_regressor.toarray()).float().to(device)
    elif not isinstance(j_regressor, torch.Tensor):
        j_regressor = torch.tensor(j_regressor, dtype=torch.float32, device=device)

    # Compute neck joint in deformed FLAME space
    neck_joint = j_regressor[neck_joint_idx] @ flame_deformed_verts  # (3,)
    neck_z = neck_joint[2].item()

    print(f"Neck joint position: {neck_joint.cpu().numpy()}")

    # Find the lowest Z value in FLAME (bottom of the mesh)
    flame_z_min = flame_deformed_verts[:, 2].min().item()
    print(f"FLAME Z range: [{flame_z_min:.4f}, {flame_deformed_verts[:, 2].max().item():.4f}]")

    # Get ICT vertex Z coordinates
    ict_z = state.ict_model.v_pos[:, 2]  # (V_ict,)
    ict_z_min = ict_z.min().item()
    ict_z_max = ict_z.max().item()
    print(f"ICT Z range: [{ict_z_min:.4f}, {ict_z_max:.4f}]")

    # Compute smooth weight falloff based on Z-coordinate
    # - Vertices above neck_z: weight = 1.0 (head moves fully)
    # - Vertices below (neck_z - falloff_range): weight = 0.0 (torso fixed)
    # - Vertices in between: smooth interpolation

    upper_z = neck_z  # Full weight above this
    lower_z = neck_z - falloff_range  # Zero weight below this

    print(f"Weight falloff zone: Z ∈ [{lower_z:.4f}, {upper_z:.4f}]")

    # Compute weights with smooth falloff
    ict_neck_weights = torch.zeros(len(ict_z), device=device)

    # Head region (above neck): full weight
    head_mask = ict_z >= upper_z
    ict_neck_weights[head_mask] = 1.0

    # Transition region: smooth falloff using smoothstep
    transition_mask = (ict_z < upper_z) & (ict_z > lower_z)
    if transition_mask.any():
        t = (ict_z[transition_mask] - lower_z) / (upper_z - lower_z)
        # Smoothstep interpolation: 3t² - 2t³
        smooth_t = 3 * t**2 - 2 * t**3
        ict_neck_weights[transition_mask] = smooth_t

    # Torso region (below transition): zero weight
    # (already initialized to zero)

    print(f"Weight distribution:")
    print(f"  Head (weight=1.0): {head_mask.sum().item()} vertices")
    print(f"  Transition (0<w<1): {transition_mask.sum().item()} vertices")
    print(f"  Torso (weight=0.0): {(~head_mask & ~transition_mask).sum().item()} vertices")
    print(f"  Weight range: [{ict_neck_weights.min().item():.4f}, {ict_neck_weights.max().item():.4f}]")

    return ict_neck_weights, neck_joint
