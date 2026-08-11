from typing import Callable

import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

class DiffLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, do_bg: bool = True):
        super(DiffLoss, self).__init__()

        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin

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
                y_onehot = torch.zeros(x.shape, device=x.device)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            gt_diff_x = y_onehot[:, :, 1:] - y_onehot[:, :, :-1]
            gt_diff_y = y_onehot[:, :, :, 1:] - y_onehot[:, :, :, :-1]
            gt_diff_z = y_onehot[:, :, :, :, 1:] - y_onehot[:, :, :, :, :-1]

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        pred_diff_x = x[:, :, 1:] - x[:, :, :-1]
        pred_diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]
        pred_diff_z = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        
        loss = (((gt_diff_x - pred_diff_x) ** 2).mean() + ((gt_diff_y - pred_diff_y) ** 2).mean() + ((gt_diff_z - pred_diff_z) ** 2).mean()) / 3.

        return loss

class DC_SkelREC_Diff_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, diff_kwargs, weight_ce=1, weight_dice=1, weight_srec=1, weight_diff=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss, srec_class=SoftSkeletonRecallLoss):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.weight_diff = weight_diff
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = srec_class(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)
        self.diff = DiffLoss(apply_nonlin=softmax_helper_dim1, **diff_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, skel) if self.weight_srec != 0 else 0
        diff_loss = self.diff(net_output, target) if self.weight_diff != 0 else 0
        ce_loss = (self.ce(net_output, target[:, 0].long())).mean() if self.weight_ce != 0 else 0
        
        if torch.isnan(dc_loss).any():
            print(f"dc_loss is NaN")
        if torch.isnan(srec_loss).any():
            print(f"srec_loss is NaN")
        if torch.isnan(diff_loss).any():
            print(f"diff_loss is NaN")
        if torch.isnan(ce_loss).any():
            print(f"ce_loss is NaN")
        # print(f"dc_loss={dc_loss}, srec_loss={srec_loss}, ce_loss={ce_loss}")

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss + self.weight_diff * diff_loss
        return result