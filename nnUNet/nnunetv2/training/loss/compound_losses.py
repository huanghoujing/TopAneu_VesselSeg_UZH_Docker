import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss, MemoryEfficientSoftDiceLossSmoothHierarchicalWindow
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss, CE_TopK_Loss, CE_FG_Loss, CE_FG_Loss_Smooth
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_CE_FG_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_fg_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, return_dict=False, print_run_avg_resolution=None, run_avg_window_size=250, print_freq=250, print_func=print):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_fg_ce = weight_fg_ce
        self.ignore_label = ignore_label

        self.ce = CE_FG_Loss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
        self.return_dict = return_dict
        self.print_run_avg_resolution = print_run_avg_resolution
        if self.print_run_avg_resolution is not None:
            self.loss_keys = ['dice_loss', 'ce_loss', 'fg_ce_loss']
            self.loss_fifo_dict = {
                k: [] for k in self.loss_keys
            }
            self.loss_avg_value_dict = {
                k: 0 for k in self.loss_keys
            }
            self.run_avg_window_size = run_avg_window_size
            self.print_freq = print_freq
            self.print_counter = 0
            print(f"[{self.__class__.__name__}] Initialized run avg loss tracking for resolution {self.print_run_avg_resolution} "+
                  f"with window size {self.run_avg_window_size} and print freq {self.print_freq}.")
        self.print_func = print_func
    
    def update_run_avg(self, this_loss_dict):
        for k in self.loss_keys:
            x = this_loss_dict[k]
            if torch.is_tensor(x):
                x = x.detach().cpu().item()
            else:
                x = float(x)
            self.loss_fifo_dict[k].append(x)
            if len(self.loss_fifo_dict[k]) > self.run_avg_window_size:
                self.loss_fifo_dict[k].pop(0)
            self.loss_avg_value_dict[k] = sum(self.loss_fifo_dict[k]) / len(self.loss_fifo_dict[k])

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Debug net_output and target shapes and unique values
        # print(f"[{self.__class__.__name__}] net_output shape: {net_output.shape}, target shape: {target.shape}")
        # print(f"[{self.__class__.__name__}] target unique values: {torch.unique(target)}")


        if mask is not None and num_fg <= 0:
            # Create zero losses that are still connected to the computational graph
            # This is necessary to avoid GradScaler errors ("No inf checks were recorded")
            # when backward() is called on a loss not connected to model parameters
            if isinstance(net_output, (list, tuple)):
                tmp_output = net_output[0]
            else:
                tmp_output = net_output
            zero_connected = tmp_output.sum() * 0.0
            ce_loss = zero_connected
            fg_ce_loss = zero_connected
            dc_loss = zero_connected
            self.print_func(f"[{self.__class__.__name__}] WARNING: No foreground pixels, skipping loss calculation.")
        else:
            dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
            ce_loss_dict = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else {'ce_loss': 0, 'fg_ce_loss': 0}
            ce_loss, fg_ce_loss = ce_loss_dict['ce_loss'], ce_loss_dict['fg_ce_loss']
        
        loss = self.weight_ce * ce_loss + self.weight_fg_ce * fg_ce_loss + self.weight_dice * dc_loss

        if self.print_run_avg_resolution is not None:
            if list(net_output.shape[2:]) == list(self.print_run_avg_resolution):
                self.update_run_avg({
                    'dice_loss': dc_loss,
                    'ce_loss': ce_loss,
                    'fg_ce_loss': fg_ce_loss,
                })
                self.print_counter += 1
                if self.print_counter % self.print_freq == 0:
                    self.print_func(
                        f"[=][=][=][=] Run avg loss at resolution {self.print_run_avg_resolution}: "+
                        f"dice_loss={self.loss_avg_value_dict['dice_loss']:.4f}, " +
                        f"ce_loss={self.loss_avg_value_dict['ce_loss']:.4f}, " +
                        f"fg_ce_loss={self.loss_avg_value_dict['fg_ce_loss']:.4f}"
                    )
                    self.print_counter = 0
        if self.return_dict:
            return {'loss': loss, 'dice_loss': dc_loss, 'ce_loss': ce_loss, 'fg_ce_loss': fg_ce_loss}
        
        return loss

class DC_CE_FG_Smooth_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_fg_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=None, return_dict=False, print_run_avg_resolution=None, run_avg_window_size=250, print_freq=250, print_func=print):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_fg_ce = weight_fg_ce
        self.ignore_label = ignore_label

        self.ce = CE_FG_Loss_Smooth(**ce_kwargs)
        self.dc = MemoryEfficientSoftDiceLossSmoothHierarchicalWindow(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
        self.return_dict = return_dict
        self.print_run_avg_resolution = print_run_avg_resolution
        if self.print_run_avg_resolution is not None:
            self.loss_keys = ['dice_loss', 'ce_loss', 'fg_ce_loss']
            self.loss_fifo_dict = {
                k: [] for k in self.loss_keys
            }
            self.loss_avg_value_dict = {
                k: 0 for k in self.loss_keys
            }
            self.run_avg_window_size = run_avg_window_size
            self.print_freq = print_freq
            self.print_counter = 0
            print(f"[{self.__class__.__name__}] Initialized run avg loss tracking for resolution {self.print_run_avg_resolution} "+
                  f"with window size {self.run_avg_window_size} and print freq {self.print_freq}.")
        self.print_func = print_func
    
    def update_run_avg(self, this_loss_dict):
        for k in self.loss_keys:
            x = this_loss_dict[k]
            if torch.is_tensor(x):
                x = x.detach().cpu().item()
            else:
                x = float(x)
            self.loss_fifo_dict[k].append(x)
            if len(self.loss_fifo_dict[k]) > self.run_avg_window_size:
                self.loss_fifo_dict[k].pop(0)
            self.loss_avg_value_dict[k] = sum(self.loss_fifo_dict[k]) / len(self.loss_fifo_dict[k])

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Debug net_output and target shapes and unique values
        # print(f"[{self.__class__.__name__}] net_output shape: {net_output.shape}, target shape: {target.shape}")
        # print(f"[{self.__class__.__name__}] target unique values: {torch.unique(target)}")


        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss_dict = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ce_loss, fg_ce_loss = ce_loss_dict['ce_loss'], ce_loss_dict['fg_ce_loss']
        
        loss = self.weight_ce * ce_loss + self.weight_fg_ce * fg_ce_loss + self.weight_dice * dc_loss

        if self.print_run_avg_resolution is not None:
            if list(net_output.shape[2:]) == list(self.print_run_avg_resolution):
                self.update_run_avg({
                    'dice_loss': dc_loss,
                    'ce_loss': ce_loss,
                    'fg_ce_loss': fg_ce_loss,
                })
                self.print_counter += 1
                if self.print_counter % self.print_freq == 0:
                    self.print_func(
                        f"[=][=][=][=] Run avg loss at resolution {self.print_run_avg_resolution}: "+
                        f"dice_loss={self.loss_avg_value_dict['dice_loss']:.4f}, " +
                        f"ce_loss={self.loss_avg_value_dict['ce_loss']:.4f}, " +
                        f"fg_ce_loss={self.loss_avg_value_dict['fg_ce_loss']:.4f}"
                    )
                    self.print_counter = 0
        if self.return_dict:
            return {'loss': loss, 'dice_loss': dc_loss, 'ce_loss': ce_loss, 'fg_ce_loss': fg_ce_loss}
        
        return loss

class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_srec=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss, srec_class=SoftSkeletonRecallLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = srec_class(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        dc_loss = self.dc(net_output, target) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, skel) \
            if self.weight_srec != 0 else 0
        ce_loss = (self.ce(net_output, target[:, 0].long())).mean() \
            if self.weight_ce != 0 else 0
        
        if torch.isnan(dc_loss).any():
            print(f"dc_loss is NaN")
        if torch.isnan(srec_loss).any():
            print(f"srec_loss is NaN")
        if torch.isnan(ce_loss).any():
            print(f"ce_loss is NaN")
        # print(f"dc_loss={dc_loss}, srec_loss={srec_loss}, ce_loss={ce_loss}")

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result
    

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

# HOUJING
class DC_CE_topk_loss(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs, weight_ce=1, weight_topk=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_topk = weight_topk
        self.ignore_label = ignore_label

        self.dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **dice_kwargs)
        self.ce_topk = CE_TopK_Loss(**ce_kwargs)
        

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss, topk_loss = self.ce_topk(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else (0, 0)

        result = self.weight_ce * ce_loss + self.weight_topk * topk_loss + self.weight_dice * dc_loss
        return result