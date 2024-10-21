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

"""
Implementation of Neuralangelo model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.models.bakedsdf import BakedSDFFactoModel, BakedSDFModelConfig
from nerfstudio.utils import colormaps
from typing_extensions import Literal
from nerfstudio.cameras.rays import RayBundle, RaySamples
from torchtyping import TensorType
from nerfstudio.models.megasurf_config import *


@dataclass
class BakedAngeloModelConfig(BakedSDFModelConfig):
    """Neuralangelo Model Config"""

    _target: Type = field(default_factory=lambda: BakedAngeloModel)
    # TODO move to base model config since it can be used in all models
    enable_progressive_hash_encoding: bool = True
    """whether to use progressive hash encoding"""
    quick_progressive_hash_encoding: bool = True
    """whether to use quick progressive hash encoding (jump the initial step)"""
    enable_numerical_gradients_schedule: bool = True
    """whether to use numerical gradients delta schedule"""
    enable_curvature_loss_schedule: bool = True
    """whether to use curvature loss weight schedule"""
    curvature_loss_multi: float = 5e-4
    """curvature loss weight"""
    curvature_loss_warmup_steps: int = 1000
    """curvature loss warmup steps"""
    level_init: int = 4
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 5000
    """steps per level of multi-resolution hash encoding"""
    geo_surf_loss_multi: float = 0.1  # 0.01
    """geometry loss, zero-crosse position"""
    guider_geo_loss_multi: float = 0
    """Guider loss weight"""
    guider_non_occupancy_loss_multi: float = 0.00
    """Guider non occupancy loss weight"""
    guider_epsilon: float = 0.02
    """guider_region_epsilon loss weight"""
    guider_region_epsilon_loss_multi: float = 1
    """Raylen expectation loss weight"""
    raylen_expectation_loss_multi: float = 0.00
    """Guider density epsilon, t should locate in [t-eps,t+eps], epsilon belongs to (0,1]"""
    training_stage: Literal["bake_geo", "init_radia", "train_all"] = "bake_geo"
    """current training step, megasurf"""
    prop_loss_weight: Tuple[float, ...] = (0.5, 0.5, 0.5)


class BakedAngeloModel(BakedSDFFactoModel):
    """Neuralangelo model

    Args:
        config: Neuralangelo configuration to instantiate model
    """

    config: BakedAngeloModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.curvature_loss_multi_factor = 1.0

    def get_intersection_points(self,
                                ray_samples: RaySamples, sdf: torch.Tensor, normal: torch.Tensor,
                                in_image_mask: torch.Tensor = None
                                ):
        """compute intersection points

        Args:
            ray_samples (RaySamples): _description_
            sdf (torch.Tensor): _description_
            normal (torch.Tensor): _description_
            in_image_mask (torch.Tensor): we only use the rays in the range of [half_patch:h-half_path, half_patch:w-half_path]
        Returns:
            _type_: _description_
        """
        # TODO we should support different ways to compute intersections
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension

        n_rays, n_samples = ray_samples.shape
        starts = ray_samples.frustums.starts
        foreground_mask = self.get_foreground_mask(ray_samples).squeeze(-1)
        ends = ray_samples.frustums.ends

        mid_z_vals = ((starts + ends) / 2).squeeze(-1)
        #######

        sdf_d = sdf.reshape(n_rays, n_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        # print(f'sign +: {((sign < 0).min(1)[0] == 0).sum()}')
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign)) * torch.logical_and(prev_sdf > 0,
                                                                                                         next_sdf < 0) * foreground_mask[
                                                                                                                         :,
                                                                                                                         1:]
        # print(f'sign ++: {((sign < 0).min(1)[0] == 0).sum()}')
        validmask_pts0 = torch.ones([n_rays], device='cuda')
        validmask_pts0 = (validmask_pts0 * sign.sum(-1)).bool()

        idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
        tmp = torch.einsum("ab,b->ab", (sign, idx))
        prev_idx = torch.argmax(tmp, 1, keepdim=True)
        next_idx = prev_idx + 1

        # prev_inside_sphere = torch.gather(foreground_mask, 1, prev_idx)
        # next_inside_sphere = torch.gather(foreground_mask, 1, next_idx)
        # mid_inside_sphere = (0.5 * (prev_inside_sphere + next_inside_sphere) > 0.5).float()
        sdf1 = torch.gather(sdf_d, 1, prev_idx)
        sdf2 = torch.gather(sdf_d, 1, next_idx)
        z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
        z_vals2 = torch.gather(mid_z_vals, 1, next_idx)

        z_vals_sdf0 = z_vals1 + torch.abs(z_vals2 - z_vals1) * torch.abs(sdf1) / (
                    torch.abs(sdf1) + torch.abs(sdf2) + 1e-5)
        # z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        # z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        # max_z_val = torch.max(starts)
        # z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        pts_sdf0_geoneus = ray_samples.frustums.origins[:, 0, :] + ray_samples.frustums.directions[:, 0,
                                                                   :] * z_vals_sdf0  # [batch_size, 1, 3]
        # #######

        sign_matrix = torch.cat([torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1).to(sdf.device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(sdf.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

        # Define mask where a valid depth value is found
        if in_image_mask is None:
            in_image_mask = torch.ones_like(mask_sign_change, device='cuda').bool()
        mask = mask_sign_change & mask_pos_to_neg & in_image_mask

        # Get depth values and function values for the interval
        d_low = starts[torch.arange(n_rays), indices, 0][mask]
        v_low = sdf[torch.arange(n_rays), indices, 0][mask]
        n_low = normal[torch.arange(n_rays), indices, :][mask]

        indices = torch.clamp(indices + 1, max=n_samples - 1)
        d_high = starts[torch.arange(n_rays), indices, 0][mask]
        v_high = sdf[torch.arange(n_rays), indices, 0][mask]
        n_high = normal[torch.arange(n_rays), indices, :][mask]

        # linear-interpolations or run secant method to refine depth
        z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

        # make this simpler
        origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
        directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]

        intersection_points = origins + directions * z[..., None]

        # interpolate normal for simplicity so we don't need to call the model again
        points_normal = (v_low[..., None] * n_high - v_high[..., None] * n_low) / (v_low[..., None] - v_high[..., None])

        points_normal = torch.nn.functional.normalize(points_normal, dim=-1, p=2)

        # filter normals that are perpendicular to view directions
        # valid = (points_normal * directions).sum(dim=-1).abs() > 0.1
        # intersection_points = intersection_points[valid]
        # points_normal = points_normal[valid]
        # new_mask = mask.clone()
        # new_mask[mask] &= valid

        return intersection_points, points_normal, z, mask, pts_sdf0_geoneus, validmask_pts0, z_vals_sdf0

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # TODO only forward the points that are inside the sphere
        field_outputs = self.field(ray_samples)
        field_outputs[FieldHeadNames.ALPHA] = ray_samples.get_alphas(field_outputs[FieldHeadNames.DENSITY])
        intersection_points, intersection_points_normal, intersection_raylen, intersection_mask, geoneus_pts0, validmask_pts0, z_vals_sdf0 = self.get_intersection_points(
            ray_samples, field_outputs[FieldHeadNames.SDF], field_outputs[FieldHeadNames.NORMAL])

        if self.config.background_model != "none":
            field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
            "intersection_mask": intersection_mask,
            "intersection_points": intersection_points,
            "intersection_points_normal": intersection_points_normal,
            "intersection_raylen": intersection_raylen,
            "geoneus_pts0": geoneus_pts0,
            "validmask_pts0": validmask_pts0,
            "z_vals_sdf0": z_vals_sdf0
        }

        return samples_and_field_outputs

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # read the hash encoding parameters from field
        level_init = self.config.level_init
        # schedule the delta in numerical gradients computation
        num_levels = self.field.num_levels
        max_res = self.field.max_res
        base_res = self.field.base_res
        growth_factor = self.field.growth_factor

        steps_per_level = self.config.steps_per_level

        init_delta = 1. / base_res
        end_delta = 1. / max_res

        # compute the delta based on level
        if self.config.enable_numerical_gradients_schedule:
            def set_delta(step):
                delta = 1. / (base_res * growth_factor ** (step / steps_per_level))
                delta = max(1. / (4. * max_res), delta)
                self.field.set_numerical_gradients_delta(
                    delta * 4.)  # TODO because we divide 4 to normalize points to [0, 1]

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_delta,
                )
            )

        # schedule the current level of multi-resolution hash encoding
        if self.config.enable_progressive_hash_encoding:

            if self.config.quick_progressive_hash_encoding:
                def set_mask(step):
                    # TODO make this consistent with delta schedule
                    level = max(level_init - 2,1) + int(step / steps_per_level)
                    level = max(level, level_init)
                    self.field.update_mask(level)
            else:
                def set_mask(step):
                    # TODO make this consistent with delta schedule
                    level = int(step / steps_per_level) + 1
                    level = max(level, level_init)
                    self.field.update_mask(level)



            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_mask,
                )
            )
        # schedule the curvature loss weight
        # linear warmup for 5000 steps to 5e-4 and then decay as delta
        if self.config.enable_curvature_loss_schedule:
            def set_curvature_loss_mult_factor(step):
                if step < self.config.curvature_loss_warmup_steps:
                    factor = step / self.config.curvature_loss_warmup_steps
                else:
                    delta = 1. / (base_res * growth_factor ** (
                                (step - self.config.curvature_loss_warmup_steps) / steps_per_level))
                    delta = max(1. / (max_res * 10.), delta)
                    factor = delta / init_delta

                self.curvature_loss_multi_factor = factor

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_curvature_loss_mult_factor,
                )
            )

        # TODO switch to analytic gradients after delta is small enough?

        return callbacks

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if self.training:
            # training statics
            metrics_dict["activated_encoding"] = self.field.hash_encoding_mask.mean().item()
            metrics_dict["numerical_gradients_delta"] = self.field.numerical_gradients_delta
            metrics_dict["curvature_loss_multi"] = self.curvature_loss_multi_factor * self.config.curvature_loss_multi

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if hasattr(self, 'global_step'):
            self.global_step += 1
        else:
            self.global_step = 0

        # curvature loss
        if self.training and self.config.curvature_loss_multi > 0.0:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs['field_outputs'][FieldHeadNames.SDF]
            sourounding_sdf = outputs['field_outputs']["sampled_sdf"]

            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))

            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = torch.abs(
                curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor

        if self.training and True:
            if hasattr(self, 'countstep'):
                self.countstep += 1
                if self.countstep == megasurf_conf['baked_end'] and self.megasurf_flag:
                    for name, param in self.field.named_parameters():
                        param.requires_grad = megasurf_fg_init[name]
                    for name, param in self.field_background.named_parameters():
                        param.requires_grad = megasurf_bg_init[name]
                    for name, param in self.proposal_networks.named_parameters():
                        param.requires_grad = megasurf_prop_init[name]
                if self.countstep == megasurf_conf['init_radiance_end'] and self.megasurf_flag:
                    for name, param in self.field.named_parameters():
                        param.requires_grad = megasurf_fg_ori[name]
                    for name, param in self.field_background.named_parameters():
                        param.requires_grad = megasurf_bg_ori[name]
                    for name, param in self.proposal_networks.named_parameters():
                        param.requires_grad = megasurf_prop_ori[name]
                    self.megasurf_flag = False
                #
                # if self.countstep >= 190000 and self.countstep < 190200:
                #     # raylen_cuda = batch["filtered_raylen"].to(self.device).reshape(-1,1)
                #     # raylen_intersection = outputs['intersection_raylen']
                #     # origins = outputs['ray_samples'].frustums.origins[:,0,:]
                #     # directions = outputs['ray_samples'].frustums.directions[:,0,:]
                #     # pcd_points = origins + directions * raylen_cuda #
                #     # pcd_points = pcd_points[(raylen_cuda > 0)[:,0]]
                #     # dpcd_points = origins + directions * outputs['depth']
                #     # # self.testpcd.append(pcd_points)
                #     # self.geopcd.append(outputs['geoneus_pts0'][outputs['validmask_pts0']])
                #     # self.intersectionpcd.append(outputs['intersection_points'])
                #     # self.depthpcd.append(dpcd_points)
                #     if self.countstep == 190100:
                #
                #
                #         self.countstep = 1000000
                #         print()
            else:
                self.countstep = 0
                self.megasurf_flag = True
                # change to baked mode
                for name, param in self.field.named_parameters():
                    param.requires_grad = megasurf_fg_baked[name]
                for name, param in self.field_background.named_parameters():
                    param.requires_grad = megasurf_bg_baked[name]
                for name, param in self.proposal_networks.named_parameters():
                    param.requires_grad = megasurf_prop_baked[name]

        if self.training:
            ## megasurf loss
            raylen_cuda = batch["filtered_raylen"].to(self.device).reshape(-1, 1)
            rayseg_max_flag = \
            (outputs["ray_samples"].frustums.starts * self.get_foreground_mask(outputs["ray_samples"])).max(1)[
                0] > raylen_cuda
            raylen_cuda[torch.logical_not(rayseg_max_flag)] = -1

            if self.megasurf_flag:
                testloss = {}
                loss_dict["exact_surface_loss"] = self.exact_surface_loss(outputs,
                                                                          raylen_cuda) * self.config.geo_surf_loss_multi
                # loss_dict["exact_surface_all_loss"] = self.exact_surface_loss_all(outputs,raylen_cuda) * self.config.geo_surf_loss_multi
                # testloss["guider_expectation_loss"] = self.guider_expectation_loss(outputs, raylen_cuda) * self.config.guider_geo_loss_multi
                loss_dict["guider_region_epsilon_loss"] = self.guider_region_epsilon_loss(outputs,
                                                                                          raylen_cuda) * self.config.guider_region_epsilon_loss_multi
            # loss_dict["non_occupancy_loss"] = self.non_occupancy_loss(outputs, raylen_cuda) * self.config.guider_non_occupancy_loss_multi
            # testloss["raylen_expectation_loss"] = self.raylen_expectation_loss(outputs, raylen_cuda) * self.config.raylen_expectation_loss_multi * max(1 - self.global_step/30000,0)
            if 'rgb_loss' in loss_dict and self.countstep < megasurf_conf['baked_end'] and self.megasurf_flag:
                loss_dict['rgb_loss'] = loss_dict['rgb_loss'] * 0
                loss_dict['interlevel_loss'] = loss_dict['interlevel_loss'] * 0
        return loss_dict

    def guider_region_epsilon_loss(self, outputs, raylen_cuda):
        valid = raylen_cuda > 0
        if valid.sum() > 0:
            low_raylen = raylen_cuda - self.config.guider_epsilon
            high_raylen = raylen_cuda + self.config.guider_epsilon
            prop_loss = 0
            element_sum = 0
            for idx in range(len(outputs['ray_samples_list'])):
                pos_mask = torch.logical_and(
                    outputs['ray_samples_list'][idx].frustums.starts[valid.squeeze(-1)] >= low_raylen[
                        valid, None, None], \
                    outputs['ray_samples_list'][idx].frustums.ends[valid.squeeze(-1)] <= high_raylen[valid, None, None])

                valid_ray = pos_mask.sum(1) > 0

                if pos_mask.sum() > 0:
                    element_sum += pos_mask.sum()
                    ray_region_weight_sum = \
                    torch.sum(outputs['weights_list'][idx][valid.squeeze(-1)] * pos_mask, dim=1)[valid_ray]
                    # ray_pre_region_weight_sum = torch.sum(outputs['weights_list'][idx][valid.squeeze(-1)] * torch.logical_not(pos_mask_pre), dim = 1)[valid_ray]
                    # prop_loss += (torch.mean(1 - ray_region_weight_sum) + torch.mean(ray_pre_region_weight_sum)) * self.config.prop_loss_weight[idx]
                    prop_loss += torch.mean(1 - ray_region_weight_sum) * self.config.prop_loss_weight[idx]

                else:
                    continue

            if element_sum > 0:
                return prop_loss
            else:
                return prop_loss
        else:
            return 0

    # expectation loss
    def guider_expectation_loss(self, outputs, raylen_cuda):
        valid = raylen_cuda > 0
        if valid.sum() > 0:
            exp_loss = torch.mean(torch.tanh(torch.abs(outputs['depth'] - raylen_cuda)[valid])) * \
                       self.config.prop_loss_weight[-1]
            for i in range(len(outputs['ray_samples_list']) - 1):
                exp_loss += torch.mean(torch.tanh(torch.abs(outputs[f'prop_depth_{i}'] - raylen_cuda)[valid])) * \
                            self.config.prop_loss_weight[i]
            return exp_loss
        else:
            return 0

    # raylen expectation loss
    def raylen_expectation_loss(self, outputs, raylen_cuda):
        valid = raylen_cuda > 0.01
        if valid.sum() > 0:
            exp_loss = self.smooth_l1_loss(outputs['depth'][valid], raylen_cuda[valid], beta=0.2, reduction='sum')
            # exp_loss = torch.mean(torch.tanh(torch.abs(outputs['depth'] - raylen_cuda)[valid]))
            return exp_loss
        else:
            return 0

    # exact surface loss
    def exact_surface_loss(self, outputs, raylen_cuda):
        valid = raylen_cuda * outputs['validmask_pts0'].unsqueeze(-1) > 0
        if valid.sum() > 0:
            surf_geoloss = self.smooth_l1_loss(outputs['z_vals_sdf0'].reshape(-1, 1)[valid], raylen_cuda[valid],
                                               beta=0.2, reduction='sum') / (valid.sum() + 1)
            return surf_geoloss
        else:
            return 0

    def exact_surface_loss_all(self, outputs, raylen_cuda):
        valid = raylen_cuda * outputs['validmask_pts0'].unsqueeze(-1) > 0
        inmask_loss = 1 - (outputs['validmask_pts0'][(raylen_cuda > 0)[:, 0]]).sum() / ((raylen_cuda > 0).sum() + 1)
        if valid.sum() > 0:
            surf_geoloss = self.smooth_l1_loss(outputs['z_vals_sdf0'].reshape(-1, 1)[valid], raylen_cuda[valid],
                                               beta=0.3, reduction='sum')  # / (valid.sum() + 1)
            return surf_geoloss + inmask_loss
        else:
            return inmask_loss

    # "geoneus_pts0": geoneus_pts0,
    # "validmask_pts0": validmask_pts0,
    # "z_vals_sdf0": z_vals_sdf0

    # non occupancy loss
    def non_occupancy_loss(self, outputs, raylen_cuda):
        valid = raylen_cuda > 0
        if valid.sum() > 0:
            low_raylen = raylen_cuda - self.config.guider_epsilon
            non_occupancy_loss = 0
            element_sum = 0
            for idx in range(len(outputs['ray_samples_list'])):
                pos_mask = outputs['ray_samples_list'][idx].frustums.starts[valid.squeeze(-1)] <= low_raylen[
                    valid, None, None]
                valid_ray = pos_mask.sum(1) > 0
                # 2024-09-07_223217
                ray_region_weight_sum = torch.sum(outputs['weights_list'][idx][valid.squeeze(-1)] * pos_mask,
                                                  dim=1).sum() / valid.sum()
                non_occupancy_loss += ray_region_weight_sum * self.config.prop_loss_weight[idx]
                ## another test:
                # if pos_mask.sum() > 0:
                #     element_sum += pos_mask.sum()
                #     ray_region_weight_sum = torch.sum(outputs['weights_list'][idx][valid.squeeze(-1)] * pos_mask, dim = 1)[valid_ray]
                #     ray_out_region_weight_sum = torch.sum(outputs['weights_list'][idx][valid.squeeze(-1)] * torch.logical_not(pos_mask), dim = 1)[valid_ray]
                #     prop_loss += torch.mean(1 - ray_region_weight_sum) + torch.mean(ray_out_region_weight_sum) * self.config.prop_loss_weight[idx]
                # else:
                #     continue

            if element_sum > 0:
                return non_occupancy_loss
            else:
                return non_occupancy_loss
        else:
            return 0

        # valid = raylen_cuda * outputs['intersection_mask'][:,None] > 0
        # if valid.sum() > 0:
        #     pos_mask = outputs["ray_samples"].frustums.ends[valid.squeeze(-1)] <= (raylen_cuda - self.config.guider_epsilon)[valid, None, None]
        #     valid_ray = pos_mask.sum(1) > 0
        #     if pos_mask.sum() > 0:
        #         ray_region_weight_sum = torch.sum(outputs['weights'][valid.squeeze(-1)] * pos_mask, dim=1)[
        #             valid_ray]
        #         nooc_loss = torch.mean(ray_region_weight_sum)
        #         return nooc_loss
        #     else:
        #         return 0
        # else:
        #     return 0

    def get_foreground_mask(self, ray_samples: RaySamples) -> TensorType:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
        """
        # inside_shpere_mask = (ray_samples.frustums.get_start_positions().norm(dim=-1, keepdim=True) < 1.0).float()
        inside_box_mask = (
                    torch.sum(ray_samples.frustums.get_start_positions().abs() > 1, -1, keepdim=True) < 1).float()
        return inside_box_mask

    def smooth_l1_loss(self, input, target, beta=1, reduction='none', key=None):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        if key != None:
            n = key(input, target)
        else:
            n = torch.abs(input - target)
        cond = n < beta
        ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret
