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

    # save filtered ICT to check if filtering is correct
    state.ict_model.filter(["Face", "HeadNeck"])
    t3d.io.obj.write(state.ict_model.template, os.path.join(state.output_dir, 'ict_filtered.obj'))