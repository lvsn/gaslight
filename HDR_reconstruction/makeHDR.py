

import os
from copy import deepcopy
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2 as cv
import numpy as np
import torch
from ezexr import imwrite
from PIL import Image
from scipy.ndimage import fourier_gaussian, zoom

from args import get_parser
from modelHDR import DiffusionHDR
import torchvision
from glob import glob
from natsort import natsorted


parser = get_parser()
args, _ = parser.parse_known_args()

args.batch_size = 1
ckpt_path = args.ckpt_path
assert ckpt_path is not None, "Please provide a checkpoint path with --ckpt_path"
model = DiffusionHDR.load_from_checkpoint(ckpt_path, args=args, strict=False).to(
    torch.device("cuda")
)

EVs = [-2, 0, 2]


def run_model(image):
    """
    Executes an inference of the model.
    `image` should be in float [0,1]
    """
    image = 2 * (image - 0.5)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).permute(0, 3, 1, 2).to("cuda")

    dicts = model.inference(image, cfg=1)
    imgs = dicts["imgs"]

    retval = {}
    EVs_no_0 = [EV for EV in EVs if EV != 0]
    for ev, img in zip(EVs_no_0, imgs):
        retval[ev] = (
            torch.clamp((img[0] + 1) / 2, 0, 1).float().permute(1, 2, 0).cpu().numpy()
        )

    return retval


def getstack(image):
    """
    Generates a linear stack of -2EV per image from an image.
    """
    torchvision.utils.save_image(torch.from_numpy(image).permute(2, 0, 1), f"output_{0}.png")
    exposure_stack = [image.copy()]
    for run_id in range(6):
        print(f"[{run_id}] Computing EV {-2*(run_id + 1)}")
        imgs = run_model(exposure_stack[-1])
        ev0 = imgs[EVs[0]]
        ev0 = np.nan_to_num(ev0)
        exposure_stack.append(ev0)
        print(exposure_stack[-1].max())
        if exposure_stack[-1].max() < 0.9:
            print("No more saturated pixels")
            break
    else:
        print(f"> ISSUE: Still saturated after {run_id} executions")

    return exposure_stack

def getstack_plus(image):
    """
    Generates a linear stack of +2EV per image from an image.
    """
    exposure_stack = [image.copy()]
    for run_id in range(2):
        print(f"[{run_id}] Computing EV {+2*(run_id + 1)}")
        imgs = run_model(exposure_stack[-1])
        ev0 = imgs[EVs[2]]
        ev0 = np.nan_to_num(ev0)
        exposure_stack.append(ev0)
        if exposure_stack[-1].min() > 0.8:
            print("No more saturated pixels")
            break

    return exposure_stack[1:]

    
def weight(v):
    weighted = -2 * np.abs(v - 0.5) + 1
    first_threshold = 0.8
    last_threshold = 0.9
    weighted[0] = np.where(v[0] < first_threshold, 1, -5 * v[0] + 5)
    weighted[-1] = np.where(v[-1] > last_threshold, 1, weighted[-1])
    return weighted

def weight_plus(v, num_plus):
    weighted = -2 * np.abs(v - 0.5) + 1
    first_threshold = 0.8
    first_threshold_plus = 0.2
    last_threshold = 0.9
    weighted[num_plus] = np.where(v[num_plus] < first_threshold, 1, -5 * v[num_plus] + 5)
    weighted[num_plus] = np.where(v[num_plus] > first_threshold_plus, weighted[num_plus], 5 * v[num_plus])
    weighted[-1] = np.where(v[-1] > last_threshold, 1, weighted[-1])
    return weighted


def merge_all(stack):
    """Custom merging function, assuming the first element of the `stack` is
    the reference exposure (where values < 0.9 will be conserved) and
    values > 0.9 will be averaged from the other exposures using a triangle as
    weighting function.
    Exposures are assumed to go by -2EV, and in sRGB (gamma 2.2) space.
    """
    exposure_times = 2 ** (-2 * np.arange(len(stack), dtype=np.float32))

    regions = np.asarray(stack)

    # eq 6 from https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
    weighted_average = weight(regions) * (
        np.log(regions**2.2) - np.log(exposure_times)[:, None, None, None]
    )
    weighted_average[np.isnan(weighted_average)] = 0.0
    E = np.exp(np.sum(weighted_average, axis=0) / np.sum(weight(regions), axis=0))
    E[np.isnan(E)] = 0.0
    image = E
    return image, None


def merge_all_plus(stack, num_plus=2):
    """Custom merging function, assuming the first element of the `stack` is
    the reference exposure (where values < 0.9 will be conserved) and
    values > 0.9 will be averaged from the other exposures using a triangle as
    weighting function.
    Exposures are assumed to go by -2EV, and in sRGB (gamma 2.2) space.
    """
    exposure_times = 2 ** (-2 * np.arange(len(stack) - num_plus, dtype=np.float32))
    exposure_times_plus = 2 ** (2 * np.arange(num_plus+1, dtype=np.float32)[1:])
    exposure_times = np.concatenate((exposure_times_plus, exposure_times))

    regions = np.asarray(stack)

    # eq 6 from https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
    weighted_average = weight_plus(regions, num_plus) * (
        np.log(regions**2.2) - np.log(exposure_times)[:, None, None, None]
    )
    weighted_average[np.isnan(weighted_average)] = 0.0
    E = np.exp(np.sum(weighted_average, axis=0) / np.sum(weight_plus(regions, num_plus), axis=0))
    E[np.isnan(E)] = 0.0
    image = E
    return image, None

    

def resize_hdr(img, size, c):
    return np.dstack(
        [
            zoom(img[:, :, chan], size, order=1, prefilter=True, grid_mode=False)
            for chan in range(c)
        ]
    )

if __name__ == "__main__":
    images = []
    
    assert args.out_dir is not None
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.image is not None:
        images.append(args.image)
    elif args.images_dir is not None:
        images = natsorted(glob(os.path.join(args.images_dir, "*." + args.ext)))
        print(f"Found {len(images)} images to process")
    else:
        raise ValueError("No image to process")
    
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        
        original_image = (np.asarray(image).copy().astype(np.float32) / 255.0)
        h, w, c = original_image.shape
        image = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS))
        image = image.astype(np.float32) / 255.0

        stack = getstack(image)
        if args.predict_ev_plus:
            stack_plus = getstack_plus(image)
            stack = stack_plus + stack
        
        if args.save_stacks:
            os.makedirs(os.path.join(args.out_dir, "stacks"), exist_ok=True)
            stacked_stack = torch.cat([torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) for x in stack], dim=0)
            torchvision.utils.save_image(stacked_stack, os.path.join(args.out_dir, "stacks", os.path.basename(image_path)[:-4] + ".png"))
        
        #Merge all in orig scale
        #Using different weights for first image EV0
        if args.predict_ev_plus:
            stack[len(stack_plus)] = original_image
            for i in range(len(stack)):
                if i == len(stack_plus):
                    continue
                stack[i] = resize_hdr(stack[i], (h / image.shape[0], w / image.shape[1]), 3)
            
            output, mask = merge_all_plus(stack)
        else:
            stack[0] = original_image
            for i in range(1, len(stack)):
                stack[i] = resize_hdr(stack[i], (h / image.shape[0], w / image.shape[1]), 3)
            
            output, mask = merge_all(stack)
        
        if np.abs(output).max() > 65504:
            print(f"Could not be contained in float16, min/max: {output.min()}/{output.max()}")
        else:
            output = output.astype(np.float16)
        
        out_path = os.path.join(args.out_dir, os.path.basename(image_path)[:-4] + ".exr")
        imwrite(out_path, output)

    root = Path().resolve()
