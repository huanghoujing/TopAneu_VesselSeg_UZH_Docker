from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import numpy as np

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, ignore_index: int = None):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.ignore_index = ignore_index

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                if self.ignore_index is not None:
                    y_tmp = y.clone()
                    y_tmp[y_tmp == self.ignore_index] = 0
                    y_onehot.scatter_(1, y_tmp.long(), 1)
                else:
                    y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

class MemoryEfficientSoftDiceLossSmoothHierarchicalWindow(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, ignore_index: int = None, fg_smooth_dilation_iterations: int = 0, temperature: float = 1.0,
                 window_sizes: tuple = None, window_weights: tuple = None, global_weight: float = 1.0,
                 min_window_fg_ratio: float = 0.0):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        Applies Gaussian smoothing to the ground truth labels before computing dice.
        Temperature > 1 makes predictions softer (less confident), < 1 makes them sharper.
        
        Hierarchical Window Dice (optional):
            window_sizes: Tuple of window sizes for hierarchical local dice, e.g. (8, 16, 32).
                         If None, only global dice is computed.
            window_weights: Weights for each window scale. If None, uniform weighting.
            global_weight: Weight for global dice when combining with window dice.
                          Final loss = global_weight * global_dice + (1 - global_weight) * window_dice
                          Set to 0.0 for pure window dice, 1.0 for pure global dice.
            min_window_fg_ratio: Minimum foreground ratio in a window to include it.
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.ignore_index = ignore_index
        self.fg_smooth_dilation_iterations = fg_smooth_dilation_iterations
        self.temperature = temperature
        
        # Hierarchical window dice parameters
        self.window_sizes = window_sizes
        self.global_weight = global_weight
        self.min_window_fg_ratio = min_window_fg_ratio
        
        if window_sizes is not None:
            if window_weights is None:
                window_weights = tuple([1.0] * len(window_sizes))
            assert len(window_weights) == len(window_sizes), "window_weights must match window_sizes length"
            self.register_buffer('window_weights', torch.tensor(window_weights, dtype=torch.float32))
        else:
            self.window_weights = None

    def _compute_windowed_dice(self, x: torch.Tensor, y_onehot: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Compute Dice scores for non-overlapping windows using average pooling.
        Returns average Dice across all valid windows. Shape: (B, C)
        """
        is_3d = x.ndim == 5
        
        # Use average pooling to compute local sums efficiently
        if is_3d:
            pool = nn.AvgPool3d(kernel_size=window_size, stride=window_size, ceil_mode=True)
            window_volume = window_size ** 3
        else:
            pool = nn.AvgPool2d(kernel_size=window_size, stride=window_size, ceil_mode=True)
            window_volume = window_size ** 2
        
        # Compute windowed sums: shape becomes (B, C, D', H', W') or (B, C, H', W')
        with torch.no_grad():
            sum_gt_windowed = pool(y_onehot) * window_volume
        
        sum_pred_windowed = pool(x) * window_volume
        intersect_windowed = pool(x * y_onehot) * window_volume
        
        # Compute Dice per window
        dice_per_window = (2 * intersect_windowed + self.smooth) / (
            torch.clip(sum_gt_windowed + sum_pred_windowed + self.smooth, 1e-8)
        )
        
        # Optionally mask out windows with too little foreground
        if self.min_window_fg_ratio > 0:
            with torch.no_grad():
                fg_ratio = sum_gt_windowed / (window_volume + 1e-8)
                valid_mask = (fg_ratio >= self.min_window_fg_ratio).float()
                valid_mask = torch.maximum(valid_mask, (sum_gt_windowed > 0).float())
        else:
            valid_mask = torch.ones_like(dice_per_window)
        
        # Average Dice across windows
        spatial_dims = tuple(range(2, dice_per_window.ndim))
        weighted_dice = (dice_per_window * valid_mask).sum(dim=spatial_dims)
        num_valid = valid_mask.sum(dim=spatial_dims).clamp(min=1)
        
        return weighted_dice / num_valid  # Shape: (B, C)

    def forward(self, x, y, loss_mask=None):
        # Apply temperature scaling before nonlinearity
        if self.temperature != 1.0:
            x = x / self.temperature
        
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                if self.ignore_index is not None:
                    y_tmp = y.clone()
                    y_tmp[y_tmp == self.ignore_index] = 0
                    y_onehot.scatter_(1, y_tmp.long(), 1)
                else:
                    y_onehot.scatter_(1, y.long(), 1)

            # Apply Gaussian smoothing to the one-hot labels
            if self.fg_smooth_dilation_iterations > 0:
                num_classes = x.shape[1]
                kernel_size = 2 * self.fg_smooth_dilation_iterations + 1
                # sigma chosen so that ~99.7% of the Gaussian is within the kernel (3-sigma rule)
                sigma = self.fg_smooth_dilation_iterations / 3.0
                sigma = max(sigma, 1e-6)
                
                # Create 1D Gaussian kernel
                coords = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - self.fg_smooth_dilation_iterations
                gaussian_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
                
                if x.ndim == 5:  # 3D case: B, C, D, H, W
                    # Create 3D Gaussian kernel via outer products
                    gaussian_kernel = gaussian_1d.view(-1, 1, 1) * gaussian_1d.view(1, -1, 1) * gaussian_1d.view(1, 1, -1)
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize to sum to 1
                    # Expand for group conv (each channel independently)
                    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size).expand(num_classes, 1, -1, -1, -1)
                    y_onehot = torch.nn.functional.conv3d(y_onehot, gaussian_kernel, padding=self.fg_smooth_dilation_iterations, groups=num_classes)
                else:  # 2D case: B, C, H, W
                    gaussian_kernel = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize to sum to 1
                    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).expand(num_classes, 1, -1, -1)
                    y_onehot = torch.nn.functional.conv2d(y_onehot, gaussian_kernel, padding=self.fg_smooth_dilation_iterations, groups=num_classes)
                
                # Re-normalize so each pixel sums to 1
                y_onehot = y_onehot / (y_onehot.sum(dim=1, keepdim=True) + 1e-8)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        # Global dice
        dc_global = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        dc_global = dc_global.mean()

        # Hierarchical window dice (if enabled)
        if self.window_sizes is not None and self.global_weight < 1.0:
            # Apply loss mask for window computation
            x_masked = x * loss_mask if loss_mask is not None else x
            y_masked = y_onehot * loss_mask if loss_mask is not None else y_onehot
            
            scale_dices = []
            for window_size in self.window_sizes:
                min_spatial = min(x.shape[2:])
                if window_size > min_spatial:
                    continue
                dice_at_scale = self._compute_windowed_dice(x_masked, y_masked, window_size)
                scale_dices.append(dice_at_scale)
            
            if len(scale_dices) > 0:
                # Stack scales: (num_scales, B, C)
                scale_dices = torch.stack(scale_dices, dim=0)
                num_scales_used = len(scale_dices)
                weights = self.window_weights[:num_scales_used].to(scale_dices.device)
                weights = weights / weights.sum()
                
                # Weighted average across scales -> (B, C)
                dc_window = (scale_dices * weights.view(-1, 1, 1)).sum(dim=0)
                
                if self.batch_dice:
                    if self.ddp:
                        dc_window = AllGatherGrad.apply(dc_window).mean(0)
                    dc_window = dc_window.mean(0)
                
                dc_window = dc_window.mean()
                
                # Combine global and window dice
                dc = self.global_weight * dc_global + (1.0 - self.global_weight) * dc_window
            else:
                dc = dc_global
        else:
            dc = dc_global

        return -dc


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
    
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec


class WeightedMemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, weight: torch.Tensor = None):
        """
        In the official DICE, `dc.mean()` averages across Batch and Class dimension.
        If some class has fewer occurance in a batch, it will be dominated by other classes.
        This imbalance is w.r.t. the number of samples per class.
        If we first average DICE within each class, then average across classes, it alleviates the imbalance.
        Upon that, we can even further assign weights for classes to make a stronger modulation.
        
        weight: Tensor with shape [n_classes]. Make it the same shape according to `do_bg`.
            E.g. if `do_bg=False`, then `weight` should not include background. Vice versa.
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.register_buffer('weight', weight)

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect) / torch.clip(sum_gt + sum_pred, 1e-8)
        
        # result shape [n_classes]
        dc = dc.sum(dim=0) / torch.clip(sum_gt.sum(dim=0), 1e-8)
        
        if self.weight is not None:
            dc = (dc * self.weight).sum() / self.weight.sum()
        else:
            dc = dc.mean()
        return -dc
    
    
class WeightedSoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, weight: torch.Tensor = None):
        """
        In the official DICE, `dc.mean()` averages across Batch and Class dimension.
        If some class has fewer occurance in a batch, it will be dominated by other classes.
        This imbalance is w.r.t. the number of samples per class.
        If we first average DICE within each class, then average across classes, it alleviates the imbalance.
        Upon that, we can even further assign weights for classes to make a stronger modulation.
        
        weight: Tensor with shape [n_classes]. Make it the same shape according to `do_bg`.
            E.g. if `do_bg=False`, then `weight` should not include background. Vice versa.
        """
        super().__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.register_buffer('weight', weight)

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
    
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)
        
        dc = inter_rec / torch.clip(sum_gt, 1e-8)
        
        # print(f"dc.shape {dc.shape}, sum_gt.shape {sum_gt.shape}")  # Both [bs, n_classes]
        # result shape [n_classes]
        dc = dc.sum(dim=0) / torch.clip(sum_gt.sum(dim=0), 1e-8)
        
        if self.weight is not None:
            dc = (dc * self.weight).sum() / self.weight.sum()
        else:
            dc = dc.mean()
        return -dc
    
    
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
