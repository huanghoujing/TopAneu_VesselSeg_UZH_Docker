import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(f"{self.__class__.__name__} {self.weight =} {self.weight.shape =}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

# HOUJING
class CE_TopK_Loss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super().__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super().forward(inp, target)
        ce_loss = res.mean()
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return ce_loss, res.mean()

class CE_TopK_FG_Loss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super().__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target, fg_mask):
        target = target[:, 0].long()
        res = super().forward(inp, target)
        ce_loss = res.mean()
        # Create graph-connected zero for when fg_ce_loss cannot be computed
        zero_connected = res.sum() * 0.0
        if fg_mask is None:
            fg_ce_loss = zero_connected
        else:
            # print(f"{res.shape = } {fg_mask.shape = }")
            if len(fg_mask.shape) > len(res.shape):
                assert fg_mask.shape[1] == 1, f"Invalid fg_mask shape {fg_mask.shape} for res shape {res.shape}"
                fg_mask = fg_mask[:, 0]
            masked = res[fg_mask]
            if masked.numel() == 0:
                fg_ce_loss = zero_connected
            else:
                fg_ce_loss = masked.mean()
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return ce_loss, res.mean(), fg_ce_loss

class CE_FG_Loss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, label_smoothing: float = 0, fg_loss_dilation_iterations: int = 0):
        super().__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)
        self.fg_loss_dilation_iterations = fg_loss_dilation_iterations

    def forward(self, inp, target):
        if len(target.shape) == len(inp.shape):
            assert target.shape[1] == 1, f"Invalid target shape {target.shape} for input shape {inp.shape}"
            target = target.squeeze(1)
        else:
            assert len(target.shape) + 1 == len(inp.shape), f"Invalid target shape {target.shape} for input shape {inp.shape}"
        
        fg_mask = (target != 0) & (target != self.ignore_index)  # ndim + 1 == inp.ndim

        # use torch conv to dilate fg_mask if needed
        # In fact, do it in one go using a big conv kernel to save time, without iterations
        if self.fg_loss_dilation_iterations > 0:
            # print(f"[=][=][=][=] Dilating fg_mask for fg loss by {self.fg_loss_dilation_iterations} iterations")
            with torch.no_grad():
                fg_mask_float = fg_mask.float().unsqueeze(1)  # add channel dim
                kernel_size = 2 * self.fg_loss_dilation_iterations + 1
                conv_kernel = torch.ones((1, 1, *([kernel_size] * (inp.ndim - 2))), device=inp.device)
                dilated = torch.nn.functional.conv3d(fg_mask_float, conv_kernel, padding=self.fg_loss_dilation_iterations)
                fg_mask_float = (dilated > 0).float()
                fg_mask = fg_mask_float[:, 0].bool()  # remove channel dim

        assert len(target.shape)+1 == len(inp.shape), f"Invalid target shape {target.shape} for input shape {inp.shape}"
        target = target.long()

        res = super().forward(inp, target)
        ce_loss = res.mean()
        # print(f"{res.shape = } {fg_mask.shape = }")
        
        masked = res[fg_mask]
        if masked.numel() == 0:
            # Use graph-connected zero to avoid GradScaler errors
            fg_ce_loss = res.sum() * 0.0
        else:
            fg_ce_loss = masked.mean()
        
        return {'ce_loss': ce_loss, 'fg_ce_loss': fg_ce_loss}

class CE_FG_Loss_Smooth(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    
    This loss extends the foreground region with a Gaussian-weighted slope.
    - fg_loss_dilation_iterations: hard dilation of the foreground mask (binary extension)
    - fg_smooth_dilation_iterations: smooth Gaussian extension beyond the (dilated) foreground
    - temperature: temperature scaling for softmax (T > 1 makes predictions softer/less confident)
    """
    def __init__(self, weight=None, ignore_index: int = -100, label_smoothing: float = 0, fg_loss_dilation_iterations: int = 0, fg_smooth_dilation_iterations: int = 0, temperature: float = 1.0):
        super().__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)
        self.fg_loss_dilation_iterations = fg_loss_dilation_iterations
        self.fg_smooth_dilation_iterations = fg_smooth_dilation_iterations
        self.temperature = temperature

    def forward(self, inp, target):
        if len(target.shape) == len(inp.shape):
            assert target.shape[1] == 1, f"Invalid target shape {target.shape} for input shape {inp.shape}"
            target = target.squeeze(1)
        else:
            assert len(target.shape) + 1 == len(inp.shape), f"Invalid target shape {target.shape} for input shape {inp.shape}"
        
        fg_mask = (target != 0) & (target != self.ignore_index)  # ndim + 1 == inp.ndim
        original_fg_mask = fg_mask  # keep original for both operations

        assert len(target.shape)+1 == len(inp.shape), f"Invalid target shape {target.shape} for input shape {inp.shape}"
        target = target.long()
        num_classes = inp.shape[1]

        # Convert target to one-hot encoding for soft label computation
        # Shape: (B, C, D, H, W) or (B, C, H, W)
        with torch.no_grad():
            # Clamp target to valid range for one-hot (handle ignore_index)
            target_clamped = target.clone()
            ignore_mask = (target == self.ignore_index)
            target_clamped[ignore_mask] = 0  # temporarily set to 0 for one-hot
            
            one_hot = torch.zeros_like(inp)
            one_hot.scatter_(1, target_clamped.unsqueeze(1), 1.0)
            
            # Zero out the one-hot for ignored pixels
            if ignore_mask.any():
                one_hot = one_hot * (~ignore_mask).unsqueeze(1).float()

        # Compute Gaussian smoothed labels
        if self.fg_smooth_dilation_iterations > 0:
            with torch.no_grad():
                kernel_size = 2 * self.fg_smooth_dilation_iterations + 1
                # sigma chosen so that ~99.7% of the Gaussian is within the kernel (3-sigma rule)
                sigma = self.fg_smooth_dilation_iterations / 3.0
                sigma = max(sigma, 1e-6)
                
                # Create 1D Gaussian kernel
                x = torch.arange(kernel_size, device=inp.device, dtype=torch.float32) - self.fg_smooth_dilation_iterations
                gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
                
                if inp.ndim == 5:  # 3D case: B, C, D, H, W
                    # Create 3D Gaussian kernel via outer products
                    gaussian_kernel = gaussian_1d.view(-1, 1, 1) * gaussian_1d.view(1, -1, 1) * gaussian_1d.view(1, 1, -1)
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize to sum to 1
                    # Expand for group conv (each channel independently)
                    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size).expand(num_classes, 1, -1, -1, -1)
                    soft_labels = torch.nn.functional.conv3d(one_hot, gaussian_kernel, padding=self.fg_smooth_dilation_iterations, groups=num_classes)
                else:  # 2D case: B, C, H, W
                    gaussian_kernel = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalize to sum to 1
                    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).expand(num_classes, 1, -1, -1)
                    soft_labels = torch.nn.functional.conv2d(one_hot, gaussian_kernel, padding=self.fg_smooth_dilation_iterations, groups=num_classes)
                
                # Re-normalize so each pixel sums to 1
                soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # No smoothing, use hard one-hot labels
            soft_labels = one_hot

        # Compute hard dilated mask independently
        if self.fg_loss_dilation_iterations > 0:
            with torch.no_grad():
                fg_mask_float = original_fg_mask.float().unsqueeze(1)  # add channel dim
                kernel_size = 2 * self.fg_loss_dilation_iterations + 1
                conv_kernel = torch.ones((1, 1, *([kernel_size] * (inp.ndim - 2))), device=inp.device)
                if inp.ndim == 5:  # 3D
                    dilated = torch.nn.functional.conv3d(fg_mask_float, conv_kernel, padding=self.fg_loss_dilation_iterations)
                else:  # 2D
                    dilated = torch.nn.functional.conv2d(fg_mask_float, conv_kernel, padding=self.fg_loss_dilation_iterations)
                hard_dilated_mask = (dilated[:, 0] > 0)  # remove channel dim
        else:
            hard_dilated_mask = original_fg_mask

        # Compute soft cross-entropy: -sum(soft_label * log_softmax(logits/T), dim=class)
        # Temperature > 1 makes predictions softer, < 1 makes them sharper
        log_softmax = torch.nn.functional.log_softmax(inp / self.temperature, dim=1)
        per_pixel_ce = -(soft_labels * log_softmax).sum(dim=1)  # (B, D, H, W) or (B, H, W)
        
        # Handle ignore_index: zero out loss for ignored pixels
        if ignore_mask.any():
            per_pixel_ce = per_pixel_ce * (~ignore_mask).float()
        
        # Use graph-connected zero to avoid GradScaler errors
        zero_connected = per_pixel_ce.sum() * 0.0
        
        # ce_loss: whole image loss with Gaussian smoothed labels
        valid_mask = ~ignore_mask
        valid_count = valid_mask.sum()
        if valid_count > 0:
            ce_loss = per_pixel_ce[valid_mask].sum() / valid_count
        else:
            ce_loss = zero_connected
        
        # fg_ce_loss: smoothed-label loss within hard-dilated region only
        fg_region_mask = hard_dilated_mask & valid_mask
        fg_count = fg_region_mask.sum()
        if fg_count > 0:
            fg_ce_loss = per_pixel_ce[fg_region_mask].sum() / fg_count
        else:
            fg_ce_loss = zero_connected
        
        return {'ce_loss': ce_loss, 'fg_ce_loss': fg_ce_loss}