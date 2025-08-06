import os
import torch
from random import randint
from gaussiansplattingInstantSplat.utils.loss_utils import l1_loss, ssim
from gaussiansplattingInstantSplat.gaussian_renderer import render, render_spherical
import sys
from gaussiansplattingInstantSplat.scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser
from gaussiansplattingInstantSplat.arguments import ModelParams, PipelineParams, OptimizationParams

import numpy as np
import torchvision

#tonemaps gamma at given exposure
def tonemap(img, EV=0):
    img = img * (2 ** EV)
    img = img.clamp(0, 1)
    img = torch.pow(img, 1/2.2)
    return img


def tonemap_np(img, EV=0):
    img = img * (2 ** EV)
    img = np.clip(img, 0, 1)
    img = np.power(img, 1/2.2)
    return img


def load_gs_hdr(images_dir, gs_ply_path, hdr=False, mask=True):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--weights', help='ignore', default=None)

    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dataset.source_path = images_dir
    #G splat config
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(gs_ply_path)
    scene = Scene(dataset, gaussians, opt=args, shuffle=False, test_mode=True, mask=mask, hdr=hdr) 
    
    bg_color = [0.1, 0.1, 0.1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    return scene, background, pipe, opt


def get_gs_hdr_pano_train_HDR(images_dir, optim_pose=True, iterations=1000, hdr=False):
    """
    imgs: 
    imgs =  N-size list of np array [(H,W,3), ...]
    focals = (N,) torch tensor 
    cams2world = (N,4,4) torch tensor 
    """

    #Argparse

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--weights', help='ignore', default=None)

    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dataset.source_path = images_dir
    #G splat config
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True, mask=True, hdr=hdr) 

    gaussians.training_setup(opt)

    bg_color = [0.1, 0.1, 0.1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    #training loop
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    

    losses_train = []
    opt.iterations = iterations
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):   
        gaussians.update_learning_rate(iteration)
        
        if optim_pose == False:
            gaussians.RT_poses.requires_grad_(False)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        index = viewpoint_cam.uid
        RT_pose = scene.gaussians.RT_poses[index] 

        bg = torch.rand((3), device="cuda")# if opt.random_background else background

        viewpoint_cam.FoVx = np.pi / 2
        viewpoint_cam.FoVy = np.pi / 2
        viewpoint_cam.compute_projection_matrix()
        viewpoint_cam.original_image = viewpoint_cam.original_image.cuda()
        viewpoint_cam.gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        render_pkg = render_spherical(viewpoint_cam, scene.gaussians, pipe, bg, RT_pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        colors_precomp = render_pkg["colors_precomp"]
        
        # Loss
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_mask = (gt_mask > 0.9).float()
        gt_image = viewpoint_cam.original_image.cuda()
        image = image * gt_mask
        gt_image = gt_image * gt_mask
                
        
        image_tmp = image.clamp(0, 1)

        gt_image_tmp = gt_image.clamp(0, 1)
        ssim_loss = 1.0 - ssim(image_tmp, gt_image_tmp)
        
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * ssim_loss

        losses_train.append([iteration, loss.item()])
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            if iteration < opt.densify_until_iter:
                #Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, colors_precomp)
                    

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if (iteration % opt.opacity_reset_interval == 0 
                    ):
                        gaussians.reset_opacity()

            if iteration % 1000 == 0:
                #render
                os.makedirs("tmp", exist_ok=True)
                out_path = os.path.join("tmp", f"render_{iteration:04d}.png")
                torchvision.utils.save_image(image_tmp, out_path)
                
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    return scene, background, pipe, opt


def get_gs_hdr_train_HDR(images_dir, optim_pose=True, iterations=1000, hdr=False):
    """
    imgs: 
    imgs =  N-size list of np array [(H,W,3), ...]
    focals = (N,) torch tensor 
    cams2world = (N,4,4) torch tensor 
    """

    #Argparse

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--weights', help='ignore', default=None)

    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dataset.source_path = images_dir
    #G splat config
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True, mask=True, hdr=hdr) 

    gaussians.training_setup(opt)

    bg_color = [0.1, 0.1, 0.1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    #training loop
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    img = scene.getTrainCameras()[0].original_image.cpu().permute(1,2,0).numpy().squeeze()
    

    losses_train = []
    opt.iterations = iterations
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):   
        gaussians.update_learning_rate(iteration)
        
        if optim_pose == False:
            gaussians.RT_poses.requires_grad_(False)
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        index = viewpoint_cam.uid
        RT_pose = scene.gaussians.RT_poses[index] 

        bg = torch.rand((3), device="cuda")

        viewpoint_cam.original_image = viewpoint_cam.original_image.cuda()
        viewpoint_cam.gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        render_pkg = render(viewpoint_cam, scene.gaussians, pipe, bg, RT_pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_mask = (gt_mask > 0.9).float()
        gt_image = viewpoint_cam.original_image.cuda()
        image = image * gt_mask
        gt_image = gt_image * gt_mask
                                    
        image_tmp = image.clamp(0, 1)
        gt_image_tmp = gt_image.clamp(0, 1)
        ssim_loss = 1.0 - ssim(image_tmp, gt_image_tmp)
        
        Ll1 = l1_loss(image, gt_image)

        loss = Ll1 + opt.lambda_dssim * ssim_loss
        losses_train.append([iteration, loss.item()])
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            if iteration < opt.densify_until_iter:
                #Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, None)
                    

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if (iteration % opt.opacity_reset_interval == 0 
                    ):
                        gaussians.reset_opacity()

            # if iteration % 1000 == 0:
            #     #render
            #     os.makedirs("tmp", exist_ok=True)
            #     out_path = os.path.join("tmp", f"render_{iteration:04d}.png")
            #     torchvision.utils.save_image(image_tmp, out_path)
                
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    return scene, background, pipe, opt
