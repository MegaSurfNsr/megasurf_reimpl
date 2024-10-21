# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
import struct
import sys
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import os

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 5000

def read_dmb(file):
    f = open(file,'rb')
    data = f.read()
    type,h,w,nb = struct.unpack('iiii',data[:16])
    datasize = h*w*nb
    z = [struct.unpack('f',data[(16+i*4):(16+4*(i+1))]) for i in range(datasize)]
    img = np.asarray(z).reshape(h,w,nb)
    return img

def depth_to_raylen(H,W,K,depth):
    depth = depth.reshape(H,W)
    u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))
    p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
    p = np.matmul(np.linalg.pinv(K), p.transpose())
    rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
    adj_cos_rev = np.dot(rays_v.transpose(), np.asarray([0., 0., 1.]))
    depth_reshape = depth.reshape(-1)
    raylen = depth_reshape / adj_cos_rev
    return raylen.reshape(H,W)

def prepare_additional_inputs(idx,confmask_file,depthacmh_file,costs_file,metadict):
    sys.stdout.write('\r' + "loading prior data idx: " + idx.__str__())
    sys.stdout.flush()
    if depthacmh_file.split('.')[-1] == 'dmb':
        depth = read_dmb(depthacmh_file) / metadict["world_to_gt"][0,0]
    else:
        depth = cv2.imread(depthacmh_file,-1)
    raylen = depth_to_raylen(metadict["H"],metadict["W"],metadict["K"],depth)
    if confmask_file is not None:
        confmask = cv2.imread(confmask_file,-1)
        raylen[confmask<1] = -1
        return {"idx": idx,
                "confmask": confmask,
                "filtered_raylen": raylen,
                "confmask_file": confmask_file,
                "depthacmh_file": depthacmh_file,
                "costs_file": costs_file,
                "threshold": metadict["threshold"]
                }
    else:
        return {"idx": idx,
                "filtered_raylen": raylen,
                "confmask_file": confmask_file,
                "depthacmh_file": depthacmh_file,
                "costs_file": costs_file,
                "threshold": metadict["threshold"]
                }



def get_single_item(idx,all_additional_data):
    assert idx == all_additional_data[idx]["idx"], "the idx doesnt match"
    if all_additional_data[idx]['confmask_file'] is not None:
        return {"confmask":all_additional_data[idx]["confmask"],"filtered_raylen":all_additional_data[idx]["filtered_raylen"]}
    else:
        return {"filtered_raylen": all_additional_data[idx]["filtered_raylen"]}


def process_megasurf_additional_data(confmask_filenames,depthacmh_filenames,costs_filenames,metadict,singlecpu=False):
    assert np.allclose(metadict['world_to_gt'][0,1],0), "for now, world_to_gt must be a similarity transfromation"
    assert metadict['world_to_gt'][0,0] > 0 and metadict['world_to_gt'][0,0] == metadict['world_to_gt'][1,1], "for now, world_to_gt must be a similarity transfromation"
    num_threads = int(os.cpu_count()) /2
    num_threads = min(num_threads, os.cpu_count() - 1)
    num_threads = max(num_threads, 1)
    fs = []
    if singlecpu:
        with ProcessPoolExecutor(max_workers=1) as executor:  # ThreadPoolExecutor ProcessPoolExecutor
            for i in range(len(confmask_filenames)):
                result = executor.submit(prepare_additional_inputs, i, confmask_filenames[i].as_posix(),
                                         depthacmh_filenames[i].as_posix(), costs_filenames[i].as_posix(), metadict)
                fs.append(result)
        # executor = ThreadPoolExecutor(max_workers=os.cpu_count()*4)
        # executor = ProcessPoolExecutor(max_workers=os.cpu_count()*2)  # exit with an exception while the function works well, try to use threadpoolexecutor
        for i in range(len(fs)):
            fs[i] = fs[i].result()
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor: # ThreadPoolExecutor
            for i in range(len(confmask_filenames)):
                try:
                    result = executor.submit(prepare_additional_inputs, i, confmask_filenames[i].as_posix(),
                                              depthacmh_filenames[i].as_posix(), costs_filenames[i].as_posix(), metadict)
                except:
                    result = executor.submit(prepare_additional_inputs, i, confmask_filenames[i],
                                              depthacmh_filenames[i].as_posix(), costs_filenames[i], metadict)
                fs.append(result)
        # executor = ThreadPoolExecutor(max_workers=os.cpu_count()*4)
        # executor = ProcessPoolExecutor(max_workers=os.cpu_count()*2)  # exit with an exception while the function works well, try to use threadpoolexecutor

        for i in range(len(fs)):
            fs[i] = fs[i].result()
        print('data loaded')
    return fs

@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 1
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = False # True
    """Whether to center the poses."""
    auto_scale_poses: bool = False # True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.94
    """The percent of images to use for training. The remaining images are for eval."""
    use_all_train_images: bool = True
    """Whether to use all images for training. If True, all images are used for training."""
    depth_filter_thred: float = 0.2
    """The threshold to filter the depth map according to the cost."""


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements
        assert self.config.downscale_factor == 1, "config.downscale_factor != 1, not implemented"
        # assert self.downscale_factor == 1, "downscale_factor != 1, not implemented"

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        confmask_filenames = []
        depthacmh_filenames = []
        costs_filenames = []
        num_skipped_image_filenames = 0

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)
            if "depth_file" in frame:
                depth_filepath = PurePath(frame["depth_file"])
                depth_fname = self._get_fname(depth_filepath, downsample_folder_prefix="depths_")
                depthacmh_filenames.append(depth_fname)
            if "cost_file" in frame:
                if len(frame["cost_file"]) == 0:
                    costs_filenames.append(None)
                else:
                    cost_filepath = PurePath(frame["cost_file"])
                    cost_fname = self._get_fname(cost_filepath, downsample_folder_prefix="costs_")
                    costs_filenames.append(cost_fname)
            if "confidence_mask" in frame:
                if len(frame["confidence_mask"]) == 0:
                    confmask_filenames.append(None)
                else:
                    confmask_filepath = PurePath(frame["confidence_mask"])
                    confmask_fname = self._get_fname(confmask_filepath, downsample_folder_prefix="confidence_mask")
                    confmask_filenames.append(confmask_fname)


        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
            if self.config.use_all_train_images:
                indices = i_all
                num_train_images = num_images
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        confmask_filenames = [confmask_filenames[i] for i in indices] if len(confmask_filenames) > 0 else []
        depthacmh_filenames = [depthacmh_filenames[i] for i in indices] if len(depthacmh_filenames) > 0 else []
        costs_filenames = [costs_filenames[i] for i in indices] if len(costs_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        ######################
        # change depth to ray length
        K = np.eye(3)
        if fx_fixed:
            K[0,0] = meta["fl_x"]
            K[1,1] = meta["fl_y"]
            K[0,2] = meta["cx"]
            K[1,2] = meta["cy"]
        else:
            K[0,0] = fx[0]
            K[1,1] = fy[0]
            K[0,2] = cx[0]
            K[1,2] = cy[0]
            print(f'warning! use same K! {K}')

        # megasurf
        results = process_megasurf_additional_data(confmask_filenames,depthacmh_filenames,costs_filenames,{"threshold":self.config.depth_filter_thred,"K":K,"H":meta["h"],"W":meta["w"],"world_to_gt" : np.array(meta["world_to_gt"]) if "world_to_gt" in meta else np.eye(4)})
        results = sorted(results,key=lambda x: x['idx'])
        for i in range(len(results)):
            if i != results[i]['idx']:
                raise ValueError("the list is not sorted")

        additional_inputs_dict = {}
        additional_inputs_dict["megasurf_prior"] = {
            "func": get_single_item,
            "kwargs": {
                "all_additional_data":results
            },
        }
        ######




        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            metadata={"transform": transform_matrix, "scale_factor": scale_factor,
                      "world_to_gt": np.array(meta["world_to_gt"] if "world_to_gt" in meta else np.eye(4)),
                      "datatype": meta["additional_datatype"] if "additional_datatype" in meta else None},
            additional_inputs=additional_inputs_dict
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath
