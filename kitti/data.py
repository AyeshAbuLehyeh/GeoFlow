"""KITTI dataset loading utilities for GeoFlow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from . import utils
except ImportError:  # pragma: no cover - standalone execution fallback
    import utils


ROOT_DIR = "/gpfs2/scratch/aabulehy/Datasets/KITTI"

SATMAP_DIR = "satmap"
GRDIMAGE_DIR = "raw_data"
LEFT_COLOR_CAMERA_DIR = "image_02/data"
OXTS_DIR = "oxts/data"

GRD_IMG_H = 256
GRD_IMG_W = 1024
GRD_ORI_IMG_H = 375
GRD_ORI_IMG_W = 1242
NUM_WORKERS = 8

TRAIN_FILE = "./train_files.txt"
TEST1_FILE = "./test1_files.txt"
TEST2_FILE = "./test2_files.txt"


@dataclass(frozen=True)
class KITTISplits:
    train: str = TRAIN_FILE
    test1: str = TEST1_FILE
    test2: str = TEST2_FILE


def _read_lines(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_left_camera_intrinsics(root: str, day_dir: str) -> torch.Tensor:
    calib_file = os.path.join(root, GRDIMAGE_DIR, day_dir, "calib_cam_to_cam.txt")
    with open(calib_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if "P_rect_02" not in line:
                continue
            values = line.split(":", 1)[1].strip().split(" ")
            fx = float(values[0]) * GRD_IMG_W / GRD_ORI_IMG_W
            cx = float(values[2]) * GRD_IMG_W / GRD_ORI_IMG_W
            fy = float(values[5]) * GRD_IMG_H / GRD_ORI_IMG_H
            cy = float(values[6]) * GRD_IMG_H / GRD_ORI_IMG_H
            camera_k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            return torch.from_numpy(camera_k)
    raise RuntimeError(f"P_rect_02 not found in {calib_file}.")


def _load_heading(root: str, drive_dir: str, image_no: str) -> float:
    oxts_file = os.path.join(root, GRDIMAGE_DIR, drive_dir, OXTS_DIR, image_no.lower().replace(".png", ".txt"))
    with open(oxts_file, "r", encoding="utf-8") as handle:
        return float(handle.readline().split(" ")[5])


def _load_rgb_image(path: str, transform=None) -> torch.Tensor:
    with Image.open(path, "r") as image:
        rgb = image.convert("RGB")
        if transform is None:
            return TF.to_tensor(rgb)
        return transform(rgb)


def _build_transform(size: Tuple[int, int], augment: bool) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms_list = [transforms.Resize(size=size)]
    if augment:
        transforms_list.extend(
            [
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomPosterize(p=0.2, bits=4),
                transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)),
            ]
        )
    transforms_list.extend([transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    return transforms.Compose(transforms_list)


class SatGrdDataset(Dataset):
    """Training dataset for KITTI cross-view geolocalization."""

    def __init__(
        self,
        root: str,
        file: str,
        transform=None,
        shift_range_lat: float = 20,
        shift_range_lon: float = 20,
        rotation_range: float = 10,
        single_file: Optional[str] = None,
    ):
        self.root = root
        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat
        self.shift_range_meters_lon = shift_range_lon
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel
        self.rotation_range = rotation_range
        self.satmap_transform = transform[0] if transform is not None else None
        self.grdimage_transform = transform[1] if transform is not None else None
        self.grdimage_dir = GRDIMAGE_DIR
        self.satmap_dir = SATMAP_DIR

        if single_file is not None:
            self.file_name = [single_file]
        else:
            self.file_name = _read_lines(file)

    def __len__(self) -> int:
        return len(self.file_name)

    def __getitem__(self, idx: int):
        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        left_camera_k = _load_left_camera_intrinsics(self.root, day_dir)
        heading = _load_heading(self.root, drive_dir, image_no)

        sat_path = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(sat_path, "r") as sat_map:
            sat_map_original = sat_map.convert("RGB")

        left_img_path = os.path.join(self.root, self.grdimage_dir, drive_dir, LEFT_COLOR_CAMERA_DIR, image_no.lower())
        grd_img = _load_rgb_image(left_img_path, self.grdimage_transform)

        sat_rot = sat_map_original.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(
            sat_rot.size,
            Image.AFFINE,
            (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel, 0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
            resample=Image.BILINEAR,
        )

        gt_shift_x = np.random.uniform(-1, 1)
        gt_shift_y = np.random.uniform(-1, 1)
        sat_rand_shift = sat_align_cam.transform(
            sat_align_cam.size,
            Image.AFFINE,
            (1, 0, gt_shift_x * self.shift_range_pixels_lon, 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
            resample=Image.BILINEAR,
        )
        theta = np.random.uniform(-1, 1)
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(theta * self.rotation_range)
        sat_map_transformed = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        if self.satmap_transform is not None:
            sat_map_transformed = self.satmap_transform(sat_map_transformed)

        gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return (
            sat_map_transformed,
            left_camera_k,
            grd_img,
            torch.tensor(gt_corr_x, dtype=torch.float32).reshape(1),
            torch.tensor(gt_corr_y, dtype=torch.float32).reshape(1),
            torch.tensor(theta, dtype=torch.float32).reshape(1),
            file_name,
        )

    def generate_correlation_GTXY(self, gt_shift_x: float, gt_shift_y: float, gt_heading: float):
        cos = np.cos(gt_heading * self.rotation_range / 180 * np.pi)
        sin = np.sin(gt_heading * self.rotation_range / 180 * np.pi)
        gt_corr_x = -gt_shift_x * cos + gt_shift_y * sin
        gt_corr_y = gt_shift_x * sin + gt_shift_y * cos
        return gt_corr_x, gt_corr_y


class SatGrdDatasetTest(Dataset):
    """Evaluation dataset for KITTI cross-view geolocalization."""

    def __init__(self, root: str, file: str, transform=None, shift_range_lat: float = 20, shift_range_lon: float = 20, rotation_range: float = 10):
        self.root = root
        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat
        self.shift_range_meters_lon = shift_range_lon
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel
        self.rotation_range = rotation_range
        self.skip_in_seq = 2
        self.satmap_transform = transform[0] if transform is not None else None
        self.grdimage_transform = transform[1] if transform is not None else None
        self.grdimage_dir = GRDIMAGE_DIR
        self.satmap_dir = SATMAP_DIR
        self.file_name = _read_lines(file)

    def __len__(self) -> int:
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx: int):
        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(" ")
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        left_camera_k = _load_left_camera_intrinsics(self.root, day_dir)
        heading = _load_heading(self.root, drive_dir, image_no)

        sat_path = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(sat_path, "r") as sat_map:
            sat_map = sat_map.convert("RGB")

        left_img_path = os.path.join(self.root, self.grdimage_dir, drive_dir, LEFT_COLOR_CAMERA_DIR, image_no.lower())
        grd_img = _load_rgb_image(left_img_path, self.grdimage_transform)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(
            sat_rot.size,
            Image.AFFINE,
            (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel, 0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
            resample=Image.BILINEAR,
        )

        gt_shift_x = -float(gt_shift_x)
        gt_shift_y = -float(gt_shift_y)

        sat_rand_shift = sat_align_cam.transform(
            sat_align_cam.size,
            Image.AFFINE,
            (1, 0, gt_shift_x * self.shift_range_pixels_lon, 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
            resample=Image.BILINEAR,
        )

        theta = float(theta)
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(theta * self.rotation_range)
        sat_map = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return (
            sat_map,
            left_camera_k,
            grd_img,
            torch.tensor(gt_corr_x, dtype=torch.float32).reshape(1),
            torch.tensor(gt_corr_y, dtype=torch.float32).reshape(1),
            torch.tensor(theta, dtype=torch.float32).reshape(1),
            file_name,
        )

    def generate_correlation_GTXY(self, gt_shift_x: float, gt_shift_y: float, gt_heading: float):
        cos = np.cos(gt_heading * self.rotation_range / 180 * np.pi)
        sin = np.sin(gt_heading * self.rotation_range / 180 * np.pi)
        gt_corr_x = -gt_shift_x * cos + gt_shift_y * sin
        gt_corr_y = gt_shift_x * sin + gt_shift_y * cos
        return gt_corr_x, gt_corr_y


def load_train_data(batch_size: int, file_path: str, shuffle: bool = True, use_augmentation: bool = False, root: str = ROOT_DIR, num_workers: int = NUM_WORKERS):
    satmap_process_sidelength = utils.get_process_satmap_sidelength()
    satmap_transform = _build_transform((satmap_process_sidelength, satmap_process_sidelength), use_augmentation)
    grdimage_transform = _build_transform((GRD_IMG_H, GRD_IMG_W), use_augmentation)

    if use_augmentation:
        print("Training with augmentation enabled.")
    else:
        print("Training without augmentation.")

    train_set = SatGrdDataset(root=root, file=file_path, transform=(satmap_transform, grdimage_transform))
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_val_data(batch_size: int, file_path: str, root: str = ROOT_DIR, num_workers: int = NUM_WORKERS):
    satmap_process_sidelength = utils.get_process_satmap_sidelength()
    satmap_transform = _build_transform((satmap_process_sidelength, satmap_process_sidelength), augment=False)
    grdimage_transform = _build_transform((GRD_IMG_H, GRD_IMG_W), augment=False)

    val_set = SatGrdDataset(root=root, file=file_path, transform=(satmap_transform, grdimage_transform))
    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_test1_data(batch_size: int, shift_range_lat: float = 20, shift_range_lon: float = 20, rotation_range: float = 10, root: str = ROOT_DIR, num_workers: int = NUM_WORKERS):
    satmap_process_sidelength = utils.get_process_satmap_sidelength()
    satmap_transform = _build_transform((satmap_process_sidelength, satmap_process_sidelength), augment=False)
    grdimage_transform = _build_transform((GRD_IMG_H, GRD_IMG_W), augment=False)

    np.random.seed(2022)
    torch.manual_seed(2022)

    test1_set = SatGrdDatasetTest(
        root=root,
        file=TEST1_FILE,
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=shift_range_lat,
        shift_range_lon=shift_range_lon,
        rotation_range=rotation_range,
    )
    return DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False)


def load_test2_data(batch_size: int, shift_range_lat: float = 20, shift_range_lon: float = 20, rotation_range: float = 10, root: str = ROOT_DIR, num_workers: int = NUM_WORKERS):
    satmap_process_sidelength = utils.get_process_satmap_sidelength()
    satmap_transform = _build_transform((satmap_process_sidelength, satmap_process_sidelength), augment=False)
    grdimage_transform = _build_transform((GRD_IMG_H, GRD_IMG_W), augment=False)

    np.random.seed(2022)
    torch.manual_seed(2022)

    test2_set = SatGrdDatasetTest(
        root=root,
        file=TEST2_FILE,
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=shift_range_lat,
        shift_range_lon=shift_range_lon,
        rotation_range=rotation_range,
    )
    return DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False)
