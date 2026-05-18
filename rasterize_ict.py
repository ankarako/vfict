import argparse
import os
import ictft.util as util
import rasterization as raster

import torch
import numpy as np
from tqdm import tqdm
import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Rasterize ICT UV space")
    parser.add_argument("--conf", type=str, help="The path to the configuration file to load.")
    args = parser.parse_args()

    print("Starting ICT uv rasterization app")
    # read configuration file
    conf = util.conf.read_conf(args.conf)

    # create output dir
    if not os.path.exists(conf.output_dir):
        os.mkdir(conf.output_dir)
    
    # import generic neutral mesh
    filenames = os.listdir(conf.ict_dir)
    for filename in filenames:
        filepath = os.path.join(conf.ict_dir, filename)

        if filename.endswith('.obj'):
            print(f"loading: {filename}")
            v_pos, v_uvs, shapes = util.io.load_model(filepath)
            
            for shape in shapes:
                output_texture = torch.zeros([conf.out_res, conf.out_res, 3], dtype=torch.float32).cuda()
                t_pos_idx = shapes[shape]['t_pos_idx']
                t_uvs_idx = shapes[shape]['t_uvs_idx']
                for idx, (t_pos_i, t_uvs_i) in tqdm(enumerate(zip(t_pos_idx, t_uvs_idx)), total=len(t_pos_idx), desc=f"Rasterizing triangles: {shape}"):
                    output_texture = raster.rasterize_triangle(v_pos[t_pos_i], v_uvs[t_uvs_i], v_pos[t_pos_i], output_texture)

                # create folder for the current mesh
                output_dir = os.path.join(conf.output_dir, filename.replace('.obj', ''))
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                # save a png image
                # min_val = output_texture.reshape(-1, 3).min(dim=0).values
                # max_val = output_texture.reshape(-1, 3).max(dim=0).values
                out_filepath = os.path.join(output_dir, f'{shape}.exr')
                # output_texture = (output_texture - min_val) / (max_val - min_val)
                output_texture = output_texture.cpu().permute(1, 0, 2).numpy()
                imageio.imsave(out_filepath, output_texture)
    print("ICT uv rasterization app terminated.")