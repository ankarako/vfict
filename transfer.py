import t3d
import mlf
import ictft

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FLAME LBS to ICT Facekit")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    conf = mlf.util.conf.read_conf(args.conf)

    # Initialize the trasfer state
    mlf.log.info("Loading Transfer state...")
    state = ictft.state_init(**conf)
    mlf.log.info("Transfer state loaded.")

    # compute rigid transformation from flame to ICT
    rot, trans, scale = ictft.rigid_alignment_lmk(state)
    v_pos_fl_aligned = ictft.rigid_flame_to_ict(state, rot, trans, scale)

    state.flame_model.v_template = v_pos_fl_aligned

    # perform nrigid ict
    v_pos_fl_deformed = ictft.non_rigid_icp(state, num_iters=20)

    # rigid transform flame to ICT
    state.ict_model.filter(["Face", "HeadNeck"])
    t3d.io.obj.write(state.ict_model.template, os.path.join(state.output_dir, 'ict_filtered.obj'))
    flame_mesh = t3d.Mesh(v_pos_fl_deformed, t_pos_idx=state.flame_model.faces)
    t3d.io.obj.write(flame_mesh, os.path.join(state.output_dir, 'flame.obj'))