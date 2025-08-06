import os
import torch
import gs

from gaussiansplattingInstantSplat.gaussian_renderer import render

from datetime import datetime

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from ezexr import imsave

import sys
import numpy as np
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()    
    parser.add_argument("--images_dir", type=str, default=None, help="Directory to images. Assumes ldrs are in images folder")
    parser.add_argument("--reconstruct_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--dont_optim_pose", action='store_true', default=False, help="Skip HDR generation")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for optimization")
    parser.add_argument("--render_final_train_views", action='store_true', default=False, help="Activate to render the training views after optimization")
    
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args, _ = parser.parse_known_args()

    assert args.images_dir is not None, "Please provide a directory to images"
    
    sys.argv = [sys.argv[0]]

    tmpdirname = "results"
    if args.reconstruct_dir:
        tmpdirname = args.reconstruct_dir
    else:
        now = datetime.now()
        tmpdirname = os.path.join(tmpdirname, now.strftime("%Y_%m_%d_%H_%M_S"))
        
    tmpdirname_HDR = os.path.join(tmpdirname, "HDR")
    os.makedirs(tmpdirname_HDR, exist_ok=True)
    
    
    #Save args params
    with open(os.path.join(tmpdirname_HDR, "args.txt"), 'w') as file:
        file.write("Independant pred \n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
            
        
    scene, background, pipe, opt = gs.get_gs_hdr_train_HDR(args.images_dir, optim_pose=(not args.dont_optim_pose), iterations=args.n_iter, hdr=True)
    
    # Render HDR
    if args.render_final_train_views:
        for i, viewpoint_cam in enumerate(scene.getTrainCameras()):   
            index = viewpoint_cam.uid
            RT_pose = scene.gaussians.RT_poses[index,:] 

            bg = torch.rand((3), device="cuda")

            render_pkg = render(viewpoint_cam, scene.gaussians, pipe, bg, RT_pose)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            image = image.detach().permute(1,2,0).cpu().numpy()
            image = image.astype(np.float16)
            imsave(os.path.join(tmpdirname_HDR, f"rendered_ldr_{index}.exr"), image)
        
    #Save gaussians LDR
    gs_ply_path = tmpdirname_HDR + f"/gaussians.ply"
    cam_pkl_path = tmpdirname_HDR + f"/cams.pkl"
    scene.gaussians.save_ply(gs_ply_path)
    # source_path = args.images_dir
    # dest_path = tmpdirname_HDR + "/optim_colmap"
    # scene.save_colmap(source_path, dest_path)
    # scene.gaussians.save_cams(cam_pkl_path)