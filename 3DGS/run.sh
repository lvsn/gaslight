#!/bin/bash

HDR_RECONSTRUCTION="../HDR_reconstruction"

ROOT="/root/gaslight/3DGS/example/" #Full path
SCENE="hall"
EXT="png" #Only png works for now

ln -s ${ROOT}${SCENE}/images ${ROOT}${SCENE}/ldr_rgb

#RUNS IN HDR_RECONSTRUCTION REPO
original_dir=$(pwd)   # Store current directory
cd $HDR_RECONSTRUCTION  # Move to the target directory
python makeHDR.py --images_dir ${ROOT}${SCENE}/images --out_dir ${ROOT}${SCENE}/hdr --ext $EXT --ckpt_path checkpoints/gaslight_weights.ckpt --save_stacks
cd "$original_dir"     # Return to the original directory

python inference_poses_vggt.py --images_dir ${ROOT}${SCENE}

python inference_gs_HDR.py --images_dir ${ROOT}${SCENE} --reconstruct_dir ${ROOT}inference/${SCENE} --n_iter 1000
