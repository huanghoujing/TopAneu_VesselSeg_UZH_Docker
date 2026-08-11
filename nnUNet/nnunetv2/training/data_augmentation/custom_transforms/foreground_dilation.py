from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize, dilation

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

def binary_dilation_3d(input_tensor, iterations=1):
    """
    Perform 3D binary dilation using PyTorch.

    Args:
        input_tensor: (B, 1, D, H, W) torch tensor, binary (0/1)
        iterations: number of dilation iterations

    Returns:
        dilated tensor, same shape as input
    """
    # Ensure tensor is float for convolution
    input_dtype = input_tensor.dtype
    x = input_tensor.float()

    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=input_tensor.device)

    for _ in range(iterations):
        # Convolution with stride=1, padding='same'
        # F.conv3d default padding='valid', so we compute padding
        padding = tuple(k // 2 for k in kernel.shape[2:])
        x = F.conv3d(x, kernel, padding=padding)
        # Any positive value becomes 1 (binary dilation)
        x = (x > 0).float()
    
    return x.to(input_dtype)


class ForegroundDilation(BasicTransform):
    def __init__(self, n_iters: int = 2):
        super().__init__()
        self.n_iters = n_iters
    
    def apply(self, data_dict, **params):
        bin_seg = data_dict['segmentation'] > 0
        bin_seg = binary_dilation_3d(bin_seg.unsqueeze(0).float(), iterations=self.n_iters).squeeze(0).to(torch.uint8)
        data_dict["dilated_fg"] = bin_seg
        return data_dict
