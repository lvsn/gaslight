import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random

from glob import glob

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from envmap import EnvironmentMap, rotation_matrix
from imageio import imread, imsave
from torch.utils.data import ConcatDataset, DataLoader, IterableDataset

###
# Tonemapping functions
###

def luminance(img):
    return img @ [0.2126, 0.7152, 0.0722]

def HLG_OETF(E):
    a = 0.17883277
    b = 1 - (4 * a)
    c = 0.5 - a * np.log(4*a)

    return np.where(
        E <= 1/12,
        np.sqrt(3 * np.minimum(E, 1/12)),
        a * np.log(12 * np.maximum(E, 1/12) - b) + c
    )


# From https://www-old.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
def Reinhard_extended(E):
    if E.ndim == 3:
        grayscale = luminance(E)
    else:
        grayscale = E
    i, j = np.unravel_index(np.argmax(grayscale), grayscale.shape)
    E_max = E[i, j, ...]
    return (E*(1. + (E/E_max**2))) / (1. + E)


def Reinhard_luminance(E):
    grayscale = luminance(E)
    new_luminance = Reinhard_extended(grayscale)
    return E*(new_luminance/grayscale)[:, :, None]


def Gamma(img, gamma=2.2):
    img = np.power(img, 1. / gamma)
    img = np.clip(img, 0, 1)
    return img


# from https://github.com/colour-science/colour-hdri/blob/develop/colour_hdri/tonemapping/global_operators/operators.py
def Filmic(
    RGB,
    shoulder_strength: float = 0.22,
    linear_strength: float = 0.3,
    linear_angle: float = 0.1,
    toe_strength: float = 0.2,
    toe_numerator: float = 0.01,
    toe_denominator: float = 0.3,
    exposure_bias: float = 2,
    linear_whitepoint: float = 11.2,
):

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    def f(
        x: float,
        A: float,
        B: float,
        C: float,
        D: float,
        E: float,
        F: float,
    ):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    RGB = f(RGB * exposure_bias, A, B, C, D, E, F)

    return RGB * (1 / f(linear_whitepoint, A, B, C, D, E, F))


tonemapping_operators = [
    # Reinhard_extended,
    # Reinhard_luminance,
    HLG_OETF,
    Gamma,
    Filmic,
]

###
# Dataset
###

class DiffusionHDRDataset(IterableDataset):
    def __init__(self, EVs=[-2, 0, 2], target_size=512, exposed_offset=True):
        super().__init__()

        self.tonemapping_operator = tonemapping_operators[0]
        self.EVs = EVs
        self.target_size = target_size
        self.exposed_offset = exposed_offset
        
        more_panos_path = "./envmaps.txt"
        self.hdrs_paths = []
        with open(more_panos_path) as fp:
            lines = fp.readlines()
            for line in lines:
                self.hdrs_paths.append(line.strip())
        # self.dataset = MadacodeIblsDataset(local_path="data")

        self.idx = 0

    def autoexpose(self, img):
        low_thres = 1e-3
        high_thres = np.percentile(img, 95)
        mask = (
            np.sum(
                np.logical_and(img < high_thres, img > low_thres), axis=2, keepdims=True
            )
            > 0
        )
        mask = np.tile(mask, (1, 1, 3))
        px = img[mask].reshape((-1, 3))
        factor = 0.17 / np.exp(np.mean(np.log(px @ [0.2989, 0.5870, 0.1140] + 1e-4)))

        img = factor * img
        # print(f"Autoexposure pixels: {mask.sum()//3}/{img.shape[0]*img.shape[1]} ({100*mask.sum()//3/(img.shape[0]*img.shape[1]):.01f}%)")
        # print(f"Autoexposure factor: {factor:.02f} (original image: {np.exp(np.mean(np.log(px @ [0.2989, 0.5870, 0.1140] + 1e-4))):.03f})")
        return img

    def get_tonemapping_operators(self):
        return tonemapping_operators

    def set_tonemapping_operator(self, tonemapping_operator):
        self.tonemapping_operator = tonemapping_operator

    def tonemap(self, img):
        img = self.tonemapping_operator(img.copy())
        img = img * 2 - 1
        return img
        

    def __iter__(self):
        return self

    def __next__(self):
        valid = False
        while not valid:
            hdr_path = self.hdrs_paths[self.idx]
            hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)

            if hdr is None:
                print(f"Dropping {hdr_path}, couldn't load")
                self.hdrs_paths.pop(self.idx)
                if self.idx >= len(self.hdrs_paths):
                    self.idx = 0
            elif np.isnan(hdr).any():
                print(f"Dropping {hdr_path} due to nan values")
                self.hdrs_paths.pop(self.idx)
                if self.idx >= len(self.hdrs_paths):
                    self.idx = 0
            else:
                valid = True

        hdr = hdr[:, :, ::-1]

        e = EnvironmentMap(hdr, "latlong")

        azimuth = random.uniform(0, 2 * np.pi)
        elevation = random.uniform(-np.pi / 2, np.pi / 2)
        roll = random.uniform(-np.pi / 16, np.pi / 16)
        fov = random.uniform(60, 120)

        dcm = rotation_matrix(azimuth=azimuth, elevation=elevation, roll=roll)
        crop = e.project(
            vfov=fov,  # degrees
            rotation_matrix=dcm,
            ar=1.0,
            resolution=(self.target_size, self.target_size),
            projection="perspective",
            mode="normal",
        )

        exposed = self.autoexpose(crop)

        if self.exposed_offset:
            EV_offset = random.randint(-12, 4)
            exposed = exposed * 2**EV_offset


        out = {}
        self.set_tonemapping_operator(random.choice(tonemapping_operators))
        for EV in self.EVs:
            img = exposed.copy() * 2**EV
            img = np.maximum(img, 0)
            img = self.tonemap(img)
            img = np.clip(img, -1, 1)
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            out[EV] = img

        self.idx += 1
        if self.idx >= len(self.hdrs_paths):
            self.idx = 0
        return out


class DiffusionHDRDataModule(pl.LightningDataModule):
    def __init__(self, args, EVs=[-2, 0, 2], target_size=512):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor

        self.target_size = target_size
        self.EVs = EVs

    def setup(self, stage=None):

        self.dataset_train = DiffusionHDRDataset(self.EVs, self.target_size)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            # num_threads=32 if self.num_workers > 0 else 1,
            pin_memory=False,
        )



if __name__ == "__main__":
    dataset = DiffusionHDRDataset()
    for idx, elem in enumerate(dataset):
        for k, v in elem.items():
            print(idx, k, v.shape)
            img = (v + 1) / 2
            torchvision.utils.save_image(img, f"{idx:02d}_{k:02d}.png")
        if idx > 10:
            break
    import pdb; pdb.set_trace()
