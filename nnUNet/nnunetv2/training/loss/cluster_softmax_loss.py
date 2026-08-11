from typing import Callable

import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import numpy as np
import torch.nn.functional as F


class ClusterSoftmaxLoss(nn.Module):
    """For sample balanced loss, refer to
    https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    https://github.com/wildoctopus/cbloss/blob/main/cbloss/loss.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        with torch.no_grad():
            target = y.long()
            if target.ndim == 5:
                assert target.shape[1] == 1, f"It looks like target is one-hot shape = {target.shape}"
                target = target.squeeze(1)  # [b, x, y, z]

            beta = 0.9999
            n_classes = 14
            samples_per_cls = torch.concat([(target==l).flatten(start_dim=1).sum(dim=1, keepdim=True) for l in range(n_classes)], dim=1)  # [batch_size b, n_classes k]
            effective_num = 1.0 - torch.pow(beta, samples_per_cls)
            weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True) * n_classes
            weights = torch.stack([_w[_t] for _w, _t in zip(weights, target)])  # [b, x, y, z]
            
            s = list(target.shape)
            s.insert(1, n_classes)
            y_onehot = torch.zeros(s, device=x.device)
            y_onehot.scatter_(1, target.unsqueeze(1), 1)
            # print(f"x.shape: {x.shape}, target.shape: {target.shape}, y_onehot.shape: {y_onehot.shape}")

        centers = torch.stack([(x * y_onehot[:, l:l+1]).sum(dim=(2,3,4)) / y_onehot[:, l:l+1].sum(dim=(2,3,4)).clamp(min=1e-8) for l in range(n_classes)], dim=1)  # [b,k,c]
        # centers = torch.einsum('bcxyz,bkxyz->bkc', x, y_onehot) / y_onehot.sum(dim=(2,3,4)).unsqueeze(-1).clamp(min=1e-8)
        # logits = torch.einsum('bcxyz,bkc->bkxyz', F.normalize(x, p=2.0, dim=1), F.normalize(centers, p=2.0, dim=2))
        logits = torch.einsum('bcxyz,bkc->bkxyz', x, centers)
        # pred = F.softmax(logits, dim=1)
        # loss = F.cross_entropy(input=pred, target=target, reduction='mean')
        # loss = F.mse_loss(logits, y_onehot)
        loss = (F.cross_entropy(input=logits, target=target, reduction='none') * weights).mean()

        return loss

class DC_SkelREC_ClusterSM_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, lcluster_kwargs, weight_ce=1, weight_dice=1, weight_srec=1, weight_lcluster=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss, srec_class=SoftSkeletonRecallLoss):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.weight_lcluster = weight_lcluster
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = srec_class(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)
        self.lcluster = ClusterSoftmaxLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        if isinstance(net_output, dict):
            # print(f"net_output is Dict, keys: {list(net_output.keys())}")
            feature = net_output['feature']
            net_output = net_output['seg_outputs']

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, skel) if self.weight_srec != 0 else 0
        lcluster_loss = self.lcluster(feature, target) if self.weight_lcluster != 0 else 0
        ce_loss = (self.ce(net_output, target[:, 0].long())).mean() if self.weight_ce != 0 else 0
        
        if torch.isnan(dc_loss).any(): print(f"dc_loss is NaN")
        if torch.isnan(srec_loss).any(): print(f"srec_loss is NaN")
        if torch.isnan(lcluster_loss).any(): print(f"lcluster_loss is NaN")
        if torch.isnan(ce_loss).any(): print(f"ce_loss is NaN")
        # print(f"dc_loss={dc_loss}, srec_loss={srec_loss}, ce_loss={ce_loss}")

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss + self.weight_lcluster * lcluster_loss
        return result