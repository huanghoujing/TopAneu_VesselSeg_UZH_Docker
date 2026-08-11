from copy import deepcopy
from typing import Tuple, List, Union

import math
import time

import SimpleITK
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import fourier_gaussian, gaussian_filter
from torch import Tensor
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.cropping import crop_tensor


class SpatialTransformGPU(BasicTransform):
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, List[int], Tuple[int, ...]],
                 random_crop: bool,
                 p_elastic_deform: float = 0,
                 elastic_deform_scale: RandomScalar = (0, 0.2),
                 elastic_deform_magnitude: RandomScalar = (0, 0.2),
                 p_synchronize_def_scale_across_axes: float = 0,
                 p_rotation: float = 0,
                 rotation: RandomScalar = (0, 2 * np.pi),
                 p_scaling: float = 0,
                 scaling: RandomScalar = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 0,
                 bg_style_seg_sampling: bool = True,
                 mode_seg: str = 'bilinear',
                 border_mode_seg: str = "zeros",
                 center_deformation: bool = True,
                 padding_mode_image: str = "zeros",
                 device = torch.device("cpu")
                 ):
        """
        magnitude must be given in pixels!
        deformation scale is given as a paercentage of the edge length
        
        padding_mode_image: see torch grid_sample documentation. This currently applies to image and regression target 
        because both call self._apply_to_image. Can be "zeros", "reflection", "border"
        """
        super().__init__()
        self.patch_size = patch_size
        if not isinstance(patch_center_dist_from_border, (tuple, list)):
            patch_center_dist_from_border = [patch_center_dist_from_border] * len(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.random_crop = random_crop
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale  # sigma for blurring offsets, in % of patch size. Larger values mean coarser deformation
        self.elastic_deform_magnitude = elastic_deform_magnitude  # determines the maximum displacement, measured in pixels!!
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_scaling = p_scaling
        self.scaling = scaling  # larger numbers = smaller objects!
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes
        self.p_synchronize_def_scale_across_axes = p_synchronize_def_scale_across_axes
        self.bg_style_seg_sampling = bg_style_seg_sampling
        self.mode_seg = mode_seg
        self.border_mode_seg = border_mode_seg
        self.center_deformation = center_deformation
        self.padding_mode_image = padding_mode_image
        self.device = device

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1

        do_rotation = np.random.uniform() < self.p_rotation
        do_scale = np.random.uniform() < self.p_scaling
        do_deform = np.random.uniform() < self.p_elastic_deform

        if do_rotation:
            angles = [sample_scalar(self.rotation, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            angles = [0] * dim
        if do_scale:
            if np.random.uniform() <= self.p_synchronize_scaling_across_axes:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=None)] * dim
            else:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            scales = [1] * dim

        # affine matrix
        if do_scale or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, scales)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], scales)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        # elastic deformation. We need to create the displacement field here
        # we use the method from augment_spatial_2 in batchgenerators
        if do_deform:
            if np.random.uniform() <= self.p_synchronize_def_scale_across_axes:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=None, patch_size=self.patch_size)
                    ] * dim
            else:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=i, patch_size=self.patch_size)
                    for i in range(dim)
                    ]

            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, self.patch_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], patch_size=self.patch_size,
                              dim=i, deformation_scale=deformation_scales[i])
                for i in range(dim)]
            # doing it like this for better memory layout for blurring
            offsets = torch.normal(mean=0, std=1, size=(dim, *self.patch_size))

            # all the additional time elastic deform takes is spent here
            for d in range(dim):
                # fft torch, slower
                # for i in range(offsets.ndim - 1):
                #     offsets[d] = blur_dimension(offsets[d][None], sigmas[d], i, force_use_fft=True, truncate=6)[0]

                # fft numpy, this is faster o.O
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                # tmp = offsets[d].numpy().astype(np.float64)
                # gaussian_filter(tmp, sigmas[d], 0, output=tmp)
                # offsets[d] = torch.from_numpy(tmp).to(offsets.dtype)
                # print(offsets.dtype)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            spatial_dims = tuple(list(range(1, dim + 1)))
            offsets = torch.permute(offsets, (*spatial_dims, 0))
        else:
            offsets = None

        shape = data_dict['image'].shape[1:]
        if not self.random_crop:
            center_location_in_pixels = [i / 2 for i in shape]
        else:
            center_location_in_pixels = []
            for d in range(0, 3):
                mn = self.patch_center_dist_from_border[d]
                mx = shape[d] - self.patch_center_dist_from_border[d]
                if mx < mn:
                    center_location_in_pixels.append(shape[d] / 2)
                else:
                    center_location_in_pixels.append(np.random.uniform(mn, mx))
        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        img = img.to(self.device)
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates

            # torch is inconsistent. AAAAaaah
            if self.padding_mode_image == 'reflection':
                pad_mode = 'reflect'
                pad_kwargs = {}
            elif self.padding_mode_image == 'zeros':
                pad_mode = 'constant'
                pad_kwargs = {'value': 0}
            elif self.padding_mode_image == 'border':
                pad_mode = 'replicate'
                pad_kwargs = {}
            else:
                raise RuntimeError('Unknown pad mode')

            img = crop_tensor(img, [math.floor(i) for i in params['center_location_in_pixels']], self.patch_size, pad_mode=pad_mode,
                              pad_kwargs=pad_kwargs)
            return img
        else:
            grid = _create_centered_identity_grid2(self.patch_size, device=self.device)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += torch.tensor(params['elastic_offsets'], dtype=torch.float32, device=self.device)
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.tensor(params['affine'], dtype=torch.float32, device=self.device))

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if self.center_deformation and params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(img.ndim - 1)))
            else:
                mn = 0

            new_center = torch.tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])], dtype=torch.float32, device=self.device)
            grid += (new_center - mn)
            # print(f'grid sample with pad mode {self.padding_mode_image}')
            return grid_sample(img[None], _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                               mode='bilinear', padding_mode=self.padding_mode_image, align_corners=False)[0]

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        segmentation = segmentation.contiguous().to(self.device)
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            segmentation = crop_tensor(segmentation,
                                       [math.floor(i) for i in params['center_location_in_pixels']],
                                       self.patch_size,
                                       pad_mode='constant',
                                       pad_kwargs={'value': 0})
            segmentation = segmentation.to(self.device)
            return segmentation
        else:
            grid = _create_centered_identity_grid2(self.patch_size, device=self.device)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += torch.tensor(params['elastic_offsets'], dtype=torch.float32, device=self.device)
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.tensor(params['affine'], dtype=torch.float32, device=self.device))

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if self.center_deformation and params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(segmentation.ndim - 1)))
            else:
                mn = 0

            new_center = torch.tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], segmentation.shape[1:])], dtype=torch.float32, device=self.device)

            grid += (new_center - mn)
            grid = _convert_my_grid_to_grid_sample_grid(grid, segmentation.shape[1:])

            if self.mode_seg == 'nearest':
                result_seg = grid_sample(
                                segmentation[None].float(),
                                grid[None],
                                mode=self.mode_seg,
                                padding_mode=self.border_mode_seg,
                                align_corners=False
                            )[0].to(segmentation.dtype)
            else:
                result_seg = torch.zeros((segmentation.shape[0], *self.patch_size), dtype=segmentation.dtype, device=self.device)
                if self.bg_style_seg_sampling:
                    for c in range(segmentation.shape[0]):
                        # Flatten the tensor and get unique sorted elements directly on GPU
                        labels = torch.unique(segmentation[c].view(-1), sorted=True)
                        # if we only have 2 labels then we can save compute time
                        if len(labels) == 2:
                            out = grid_sample(
                                    ((segmentation[c] == labels[1]).float())[None, None],
                                    grid[None],
                                    mode=self.mode_seg,
                                    padding_mode=self.border_mode_seg,
                                    align_corners=False
                                )[0][0] >= 0.5
                            result_seg[c][out] = labels[1]
                            result_seg[c][~out] = labels[0]
                        else:
                            for i, u in enumerate(labels):
                                result_seg[c][
                                    grid_sample(
                                        ((segmentation[c] == u).float())[None, None],
                                        grid[None],
                                        mode=self.mode_seg,
                                        padding_mode=self.border_mode_seg,
                                        align_corners=False
                                    )[0][0] >= 0.5] = u
                else:
                    for c in range(segmentation.shape[0]):
                        # Flatten the tensor and get unique sorted elements directly on GPU
                        labels = torch.unique(segmentation[c].view(-1), sorted=True)
                        #torch.where(torch.bincount(segmentation.ravel()) > 0)[0].to(segmentation.dtype)
                        tmp = torch.zeros((len(labels), *self.patch_size), dtype=torch.float16, device=self.device)
                        scale_factor = 1000
                        done_mask = torch.zeros(*self.patch_size, dtype=torch.bool, device=self.device)
                        for i, u in enumerate(labels):
                            tmp[i] = grid_sample(((segmentation[c] == u).float() * scale_factor)[None, None], grid[None],
                                                 mode=self.mode_seg, padding_mode=self.border_mode_seg, align_corners=False)[0][0]
                            mask = tmp[i] > (0.7 * scale_factor)
                            result_seg[c][mask] = u
                            done_mask = done_mask | mask
                        if not torch.all(done_mask):
                            result_seg[c][~done_mask] = labels[tmp[:, ~done_mask].argmax(0)]
                        del tmp
            del grid
            return result_seg.contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = Rz @ Ry @ Rx @ S
    return RS


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S
    return RS


def _create_centered_identity_grid2(size: Union[Tuple[int, ...], List[int]], device = torch.device("cpu")) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s, device=device) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    return grid


def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Union[Tuple[int, ...], List[int]]):
    # rescale
    for d in range(len(original_shape)):
        s = original_shape[d]
        my_grid[..., d] /= (s / 2)
    my_grid = torch.flip(my_grid, (len(my_grid.shape) - 1, ))
    # my_grid = my_grid.flip((len(my_grid.shape) - 1,))
    return my_grid



if __name__ == '__main__':
    
    #################
    # with this part we can qualitatively test that the correct axes are ebing augmented. Just set one of the probs to 1 and off you go
    #################

    """
    ============================
    p_elastic_deform=0,
    p_rotation=1,
    p_scaling=0,
    ============================

    Device: cpu
    Transformed shape: (64, 60, 68)
    Average time per transformation: 0.005929689407348632 seconds

    Device: cuda:0
    Transformed shape: (64, 60, 68)
    Average time per transformation: 0.0008948326110839843 seconds

    Device: cpu
    Transformed shape: (128, 192, 192)
    Average time per transformation: 0.10765635013580323 seconds

    Device: cuda:0
    Transformed shape: (128, 192, 192)
    Average time per transformation: 0.0043716907501220705 seconds
    
    Device: cpu
    Transformed shape: (128, 256, 256)
    Average time per transformation: 0.2012350368499756 seconds

    Device: cuda:0
    Transformed shape: (128, 256, 256)
    Average time per transformation: 0.008758220672607422 seconds
    
    ============================
    p_elastic_deform=1,
    p_rotation=1,
    p_scaling=1,
    ============================

    Device: cpu
    Transformed shape: (128, 192, 192)
    Average time per transformation: 0.11614971876144409 seconds

    Device: cuda:0
    Transformed shape: (128, 192, 192)
    Average time per transformation: 0.008179898262023927 seconds
    
    """

    def eldef_scale(image, dim, patch_size):
        return 0.1

    def eldef_magnitude(image, dim, patch_size, deformation_scale):
        return 10 if dim == 2 else 0

    def rot(image, dim):
        return 45/360 * 2 * np.pi if dim == 0 else 0

    def scaling(image, dim):
        return 0.5 if dim == 0 else 1

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # shape = (1, 64, 60, 68)
    shape = (1, 128, 192, 192)
    
    # lines
    patch = torch.zeros(shape, device=device)
    patch[:, :, 10, 30] = 1
    patch[:, 50, :, 30] = 1
    patch[:, 40, 20, :] = 1

    # patch_block
    patch_block = torch.zeros(shape, device=device)
    patch_block[:, 22:42, 20:40, 24:44] = 1

    patch_line = torch.zeros(shape, device=device)
    patch_line[:, 22:24, 30:32, 10:-10] = 1
    use = patch_line

    sp = SpatialTransformGPU(
        patch_size=patch.shape[1:],
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=1,
        p_rotation=1,
        p_scaling=1,
        elastic_deform_scale=eldef_scale,
        elastic_deform_magnitude=eldef_magnitude,
        p_synchronize_def_scale_across_axes=0,
        rotation=rot,
        scaling=scaling,
        p_synchronize_scaling_across_axes=0,
        bg_style_seg_sampling=False,
        mode_seg='bilinear',
        device=device
    )

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(use[0].cpu().numpy()), 'orig.nii.gz')
    params = sp.get_parameters(image=use)

    start_time = time.time()
    n_iters = 100
    for _ in range(n_iters):
        transformed = sp._apply_to_image(use, **params)[0].cpu().numpy()
    end_time = time.time()
    print(f"Device: {device}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Average time per transformation: {(end_time - start_time) / n_iters} seconds")

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed), 'transformed.nii.gz')
