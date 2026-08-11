"""
What's special about this generic trainer:
- Loss function succeeded in TopCoW task-1-seg
- Validation only at some epochs
- Perform actual validation, metrics from whole image, no from some crops
- Metric figure plots DICE of each class, not only the mean DICE
- NOTE:
    - The training rotation augmentation is limited to 30 degrees
    - mirror_axes = (0,1)
    - Number of epochs is reduced to 200
"""
import numpy as np
import torch
from torch import autocast
from typing import Tuple, Union, List
from nnunetv2.training.loss.diff_loss import DiffLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context

from nnunetv2.training.dataloading.data_loader_2d_skel import nnUNetDataLoader2DSkel
from nnunetv2.training.dataloading.data_loader_3d_skel import nnUNetDataLoader3DSkel
from nnunetv2.training.dataloading.data_loader_3d_skel_cls_balanced_global import nnUNetDataLoader3DSkelClsBalancedGlobal, case_sampling_weight
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.skeletonization import SkeletonTransform

import os
import sys
from time import time, sleep
import warnings
import multiprocessing
from torch import distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.logging.every_nEpoch_actual_val_logger import MetricLogger
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size

from dynamic_network_architectures.architectures import unet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss
from nnunetv2.training.loss.robust_ce_loss import CE_TopK_Loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import numpy as np
import torch.nn.functional as F

# set multiprocess to spawn to avoid potential issues on some systems
multiprocessing.set_start_method('spawn', force=True)

def _rd(x, decimals=4):
    return np.round(x, decimals=decimals)


class ClusterSoftmaxLoss(nn.Module):
    """For sample balanced loss, refer to
    https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    https://github.com/wildoctopus/cbloss/blob/main/cbloss/loss.py
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, feature, target):
        with torch.no_grad():
            if target.ndim == 5:
                assert target.shape[1] == 1, f"It looks like target is one-hot shape = {target.shape}"
            # Resolution guard: the feature map may be at lower resolution than the target (e.g. stock
            # Primus' half-res pre-classifier feature). Downsample the target (nearest) to the feature's
            # spatial shape so the per-voxel clustering aligns. No-op for the UNet path and for the
            # full-res Primus head (shapes already match).
            sf = tuple(feature.shape[2:])
            if tuple(target.shape[-len(sf):]) != sf:
                t = target.float()
                added_channel = False
                if t.ndim == len(sf) + 1:  # [b, *spatial] -> [b, 1, *spatial] for interpolate
                    t = t.unsqueeze(1)
                    added_channel = True
                t = F.interpolate(t, size=sf, mode='nearest')
                target = t.squeeze(1) if added_channel else t
            target = target.long().flatten(start_dim=1)  # [b,v], v=xyz

            beta = 0.9999
            k = self.num_classes
            [b,v] = list(target.shape)
            y_onehot = torch.zeros([b,v,k], device=feature.device)
            y_onehot.scatter_(dim=2, index=target.unsqueeze(2), value=1)  # [b,v,k]
            samples_per_cls = y_onehot.sum(dim=1)  # [b,k]
            normed_y_onehot = y_onehot / samples_per_cls.unsqueeze(1).clamp(min=1e-8)
            # samples_per_cls = torch.concat([(target==l).sum(dim=1, keepdim=True) for l in range(k)], dim=1)  # [b,k]
            effective_num = 1.0 - torch.pow(beta, samples_per_cls)
            weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True) * k
            # print(f"samples_per_cls:{samples_per_cls}, weights: {weights}")
            weights = torch.gather(weights, 1, target)  # [b,v]
            
            # print(f"feature.shape: {feature.shape}, target.shape: {target.shape}, y_onehot.shape: {y_onehot.shape}")

        feature = feature.flatten(start_dim=2)  # [b,c,v]
        # If we directly use `torch.bmm(feature, y_onehot) / y_onehot.sum(xxx)`, it will generate INF values.
        centers = torch.bmm(feature, normed_y_onehot) # [b,c,k]
        # print(centers.min(), centers.max())  # make sure no INF value here
        logits = torch.bmm(centers.transpose(1,2), feature)  # [b,k,v]
        # print(logits.min(), logits.max())  # make sure no NaN value here

        # loss = F.cross_entropy(input=logits, target=target, reduction='mean')
        loss = (F.cross_entropy(input=logits, target=target, reduction='none') * weights).mean()

        return loss


class DC_SkelREC_Diff_ClusterSM_and_CE_TopK_loss(nn.Module):
    def __init__(self, ce_class_weight, n_classes_w_bg, weight_ce, weight_topk, weight_dice, weight_srec, weight_diff, weight_lcluster, dice_batch_dice=False, dice_do_bg=True):
        super().__init__()

        self.ce_topk = CE_TopK_Loss(**{'weight': None if ce_class_weight is None else torch.FloatTensor(ce_class_weight).to('cuda'),})
        self.dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **{'batch_dice': dice_batch_dice, 'smooth': 1e-5, 'do_bg': dice_do_bg, 'ddp': False})
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False})
        self.diff = DiffLoss(apply_nonlin=softmax_helper_dim1)
        self.lcluster = ClusterSoftmaxLoss(n_classes_w_bg)

        self.weight_ce = weight_ce
        self.weight_topk = weight_topk
        self.weight_dice = weight_dice
        self.weight_srec = weight_srec
        self.weight_diff = weight_diff
        self.weight_lcluster = weight_lcluster

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        if isinstance(net_output, dict):
            # print(f"net_output is Dict, keys: {list(net_output.keys())}")
            feature = net_output['feature']
            net_output = net_output['seg_outputs']

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, skel) if self.weight_srec != 0 else 0
        diff_loss = self.diff(net_output, target) if self.weight_diff != 0 else 0
        lcluster_loss = self.lcluster(feature, target) if self.weight_lcluster != 0 else 0
        ce_loss, topk_loss = self.ce_topk(net_output, target)
        
        if isinstance(dc_loss, torch.Tensor) and torch.isnan(dc_loss).any(): print(f"dc_loss is NaN")
        if isinstance(srec_loss, torch.Tensor) and torch.isnan(srec_loss).any(): print(f"srec_loss is NaN")
        if isinstance(diff_loss, torch.Tensor) and torch.isnan(diff_loss).any(): print(f"diff_loss is NaN")
        if isinstance(lcluster_loss, torch.Tensor) and torch.isnan(lcluster_loss).any(): print(f"lcluster_loss is NaN")
        if isinstance(ce_loss, torch.Tensor) and torch.isnan(ce_loss).any(): print(f"ce_loss is NaN")
        # print(f"dc_loss={dc_loss}, srec_loss={srec_loss}, ce_loss={ce_loss}")

        result = self.weight_ce * ce_loss + self.weight_topk * topk_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss + self.weight_diff * diff_loss + self.weight_lcluster * lcluster_loss
        return result

class DeepSupervisionFullPlusCEDice(nn.Module):
    """DS wrapper for DC_SkelREC_Diff_ClusterSM_and_CE_TopK_loss.
    Level 0 (full-res): full loss. Deeper (down-sampled) levels: CE + Dice only.
    weight_factors: per-level scalars (0 => skip that level)."""
    def __init__(self, full_loss, weight_factors):
        super().__init__()
        assert any(w != 0 for w in weight_factors), "At least one weight factor should be != 0.0"
        self.full_loss = full_loss            # DC_SkelREC_Diff_ClusterSM_and_CE_TopK_loss
        self.weight_factors = tuple(weight_factors)

    def forward(self, net_output, target, skel):
        feature = net_output['feature']
        seg_outputs = net_output['seg_outputs']        # list, full-res first
        w = self.weight_factors
        # level 0: full loss (reconstruct a single-output dict so cluster loss uses full-res feature)
        total = w[0] * self.full_loss({'seg_outputs': seg_outputs[0], 'feature': feature},
                                      target[0], skel)
        # deeper levels: CE + Dice only
        for i in range(1, len(seg_outputs)):
            if w[i] == 0:
                continue
            ce_loss, _ = self.full_loss.ce_topk(seg_outputs[i], target[i])
            dc_loss = self.full_loss.dc(seg_outputs[i], target[i])
            total = total + w[i] * (self.full_loss.weight_ce * ce_loss
                                    + self.full_loss.weight_dice * dc_loss)
        return total


class UNetDecoder_wF(UNetDecoder):
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return {'seg_outputs': r, 'feature': lres_input} if self.training else r
    

class Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg(nnUNetTrainer):
    # gradient clipping max-norm used in train_step; subclasses (e.g. the Primus trainer) may override
    # to 1. Class attribute so it is available even to EnvCfg subclasses that bypass this __init__.
    gradient_clip_norm = 12

    # TopBrain validation metrics (clsAvgDice/B0/F1). Class-attribute defaults so subclasses that
    # bypass this __init__ still have them; __init__ / EnvCfg may override per instance.
    to_compute_topbrain = False
    topbrain_track = 'auto'

    """HJ: I am lazy enough to pass a lot of parameters through env variables, so that I don't have to change the function signatures."""
    def __init__(self, *args, **kwargs):
        unet.UNetDecoder = UNetDecoder_wF
        nnUNetTrainer.__init__(self, *args, **kwargs)  # Directly call nnUNetTrainer.__init__

        self.enable_deep_supervision = os.environ.get('enable_deep_supervision', '0') == '1'  # default OFF, some loss functions may not support DS
        _nd = os.environ.get('num_ds_levels', None)
        self.num_ds_levels = int(_nd) if _nd is not None else None   # # of down-sampled DS levels (full-res excluded); None => all but lowest-res output
        self.sampling_probabilities = None
        self.num_images_properties_loading_threshold = 100  # HJ: Now I think it does not matter much. If it is volume data that we are caching, then it is important for RAM and speed.
        self.logger = MetricLogger()
        self.num_epochs_per_val = int(os.environ.get('num_epochs_per_val', 1000))  # Default to a very large number to effectively disable intermediate validation
        self.save_every = 1
        
        self.weight_ce=1
        self.weight_topk=float(os.environ.get('weight_topk', 0.1))          # default ON, but small to avoid noisy signal
        self.weight_dice=1
        self.weight_srec=float(os.environ.get('weight_srec', 1))            # default ON
        self.weight_diff=float(os.environ.get('weight_diff', 0))            # default OFF
        self.weight_lcluster=float(os.environ.get('weight_lcluster', 0))    # default OFF, to save VRAM. The operation on full-res feature maps is memory-consuming, especially when num_classes is large.

        self.dice_batch_dice = os.environ.get('batch_dice', self.configuration_manager.batch_dice) in ('1', True)   # Allow overwriting with env variable
        self.dice_do_bg = os.environ.get('dice_do_bg', '0') == '1'          # default OFF, as in official nnUNet

        ###########################################################################
        # For other datasets, you only need to change
        # - mirror/rotation augmentation if you will
        # - num_iterations_per_epoch, num_epochs, num_epochs_per_val, save_every
        # - and these:
        self.n_classes_w_bg = int(os.environ.get('n_classes_w_bg', 2))  # Default is dummy 1 fg + background = 2, make sure to set this according to your dataset
        ce_class_weight = os.environ.get('ce_class_weight', None)
        if ce_class_weight:
            self.ce_class_weight = [float(x) for x in ce_class_weight.split(',') if x]
            assert len(self.ce_class_weight) == self.n_classes_w_bg, f"Length of ce_class_weight ({len(self.ce_class_weight)}) must match n_classes_w_bg ({self.n_classes_w_bg})"
        else:
            self.ce_class_weight = None

        # Foreground patch-center sampling: when ON, a forced-fg sample picks the fg class FIRST (globally balanced,
        # configurable weights), then a case containing it. Foreground oversampling is always probabilistic for this
        # trainer: each sample is forced fg iff random() < oversample_foreground_percent (OVERSAMPLE_FOREGROUND_PERCENT).
        self.cls_balanced_global_sampling = os.environ.get('cls_balanced_global_sampling', '1') == '1'  # default ON
        # fg-class sampling weights have three mutually exclusive modes:
        #   - manual: fg_class_sampling_weights (comma list, len == n fg classes)
        #   - calibrated: fg_class_calibration_degree (float); baseline (degree 0) = each class's case-frequency
        #       ratio, prob ~ f_c ** (1 - degree); degree 1 = uniform, degree > 1 over-samples rare classes
        #   - uniform: neither set (default)
        fg_class_sampling_weights = os.environ.get('fg_class_sampling_weights', None)
        if fg_class_sampling_weights:
            self.fg_class_sampling_weights = [float(x) for x in fg_class_sampling_weights.split(',') if x]
            assert len(self.fg_class_sampling_weights) == self.n_classes_w_bg - 1, \
                f"Length of fg_class_sampling_weights ({len(self.fg_class_sampling_weights)}) must match the number of " \
                f"foreground classes (n_classes_w_bg - 1 = {self.n_classes_w_bg - 1})"
        else:
            self.fg_class_sampling_weights = None  # uniform over fg classes (unless calibrated below)
        fg_class_calibration_degree = os.environ.get('fg_class_calibration_degree', None)
        self.fg_class_calibration_degree = float(fg_class_calibration_degree) if fg_class_calibration_degree else None
        assert not (self.fg_class_sampling_weights is not None and self.fg_class_calibration_degree is not None), \
            "Set at most one of fg_class_sampling_weights and fg_class_calibration_degree"

        # Per-case sampling "potential" by name pattern. Default: every case weight 1. Configure groups to down-/up-
        # weight cases whose name contains given substrings, e.g. cross-modality synthetic cases. Format (variable
        # number of groups): "PAT1|PAT2:weight;PAT3:weight" -- groups separated by ';', patterns within a group by
        # '|', weight after the last ':'. The first matching group wins. Example: "CT2MR|MR2CT:0.75".
        case_sampling_weights = os.environ.get('case_sampling_weights', None)
        if case_sampling_weights:
            groups = []
            for grp in case_sampling_weights.split(';'):
                grp = grp.strip()
                if not grp:
                    continue
                pats, w = grp.rsplit(':', 1)
                patterns = [p for p in pats.split('|') if p]
                assert patterns, f"case_sampling_weights group '{grp}' has no patterns"
                groups.append((patterns, float(w)))
            self.case_sampling_weight_groups = groups if groups else None
        else:
            self.case_sampling_weight_groups = None

        self.num_cached = 32
        self.p_rotation = float(os.environ.get('p_rotation', 0.2))  # default ON, but not too frequent to save time. rotation aug is very slow when patch size is large.
        self.p_scaling = float(os.environ.get('p_scaling', 0.2))
        self.scaling = (float(os.environ.get('scaling_min', 0.7)), float(os.environ.get('scaling_max', 1.4)))
        self.segmentation_export_pool_size = int(os.environ.get('segmentation_export_pool_size', 2))  # Small pool size to avoid OOM
        self.metrics_num_processes = 4
        self.compute_hd95 = os.environ.get('compute_hd95', '0') == '1'                          # default OFF
        self.to_compute_contamination = os.environ.get('to_compute_contamination', '0') == '1'  # default OFF
        self.to_compute_topbrain = os.environ.get('to_compute_topbrain', '1') == '1'            # default ON: TopBrain clsAvgDice/B0/F1
        self.topbrain_track = os.environ.get('topbrain_track', 'auto')                          # 'auto' splits mixed CT+MR by filename
        # Distance-Weighted Error (DWE).
        self.to_compute_dwe = os.environ.get('to_compute_dwe', '0') == '1'
        self.dwe_max_dist = float(os.environ.get('dwe_max_dist', 500.0))  # Max distance (voxels) for DWE
        self.dwe_norm_by = float(os.environ.get('dwe_norm_by', 1e6))  # normalize to avoid huge summed value
        self.dwe_edt_cache_dir = None  # EDT cache defaults to <gt_segmentations>/.dwe_edt_cache.
        self.dwe_weight_fn = os.environ.get('dwe_weight_fn', 'linear')   # 'linear' | 'ones' (counts) | 'margin_linear' | 'margin_ones'
        self.dwe_weight_margin = float(os.environ.get('dwe_weight_margin', 3.0))  # margin (voxels) for dwe_weight_fn margin_linear/margin_ones
        ###########################################################################

        self.print_to_log_file(f"\n")
        self.print_to_log_file(f"enable_deep_supervision: {self.enable_deep_supervision}")
        self.print_to_log_file(f"num_ds_levels (down-sampled DS levels; None => all but lowest-res): {self.num_ds_levels}")
        self.print_to_log_file(f"num_images_properties_loading_threshold: {self.num_images_properties_loading_threshold}")
        self.print_to_log_file(f"batch_size: {self.configuration_manager.batch_size}")
        self.print_to_log_file(f"num_epochs_per_val: {self.num_epochs_per_val}")
        self.print_to_log_file(f"save_every: {self.save_every}")
        self.print_to_log_file(f"oversample_foreground_percent (per-sample fg prob, always probabilistic): {self.oversample_foreground_percent}")
        self.print_to_log_file(f"cls_balanced_global_sampling: {self.cls_balanced_global_sampling}, fg_class_sampling_weights: {self.fg_class_sampling_weights}, fg_class_calibration_degree: {self.fg_class_calibration_degree}")
        self.print_to_log_file(f"case_sampling_weight_groups: {self.case_sampling_weight_groups}")
                
        self.print_to_log_file(f"{self.num_cached = }")
        self.print_to_log_file(f"{self.p_rotation = }, {self.p_scaling = }, {self.scaling = }")
        self.print_to_log_file(f"{self.segmentation_export_pool_size = }, {self.metrics_num_processes = }")
        self.print_to_log_file(f"{self.compute_hd95 = }")
        self.print_to_log_file(f"{self.to_compute_contamination = }")
        self.print_to_log_file(f"{self.to_compute_topbrain = }, {self.topbrain_track = }")
        self.print_to_log_file(f"{self.to_compute_dwe = }, {self.dwe_max_dist = }, {self.dwe_norm_by = }, {self.dwe_weight_fn = }, {self.dwe_weight_margin = }")
        self.print_to_log_file(f"\n")
        #############################################################################

        self.print_to_log_file(f"{self.__class__.__name__} n_classes_w_bg: {self.n_classes_w_bg}, ce_class_weight: {self.ce_class_weight}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_ce: {self.weight_ce}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_topk: {self.weight_topk}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_dice: {self.weight_dice}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_srec: {self.weight_srec}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_diff: {self.weight_diff}")
        self.print_to_log_file(f"[{self.__class__.__name__}] weight_lcluster: {self.weight_lcluster}")
        
        self.print_to_log_file(f"[{self.__class__.__name__}] dice_batch_dice: {self.dice_batch_dice}")
        self.print_to_log_file(f"[{self.__class__.__name__}] dice_do_bg: {self.dice_do_bg}")
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        do_dummy_2d_data_aug = False
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        mirror_axes = os.getenv('MIRROR_AXES', (0,1))
        if isinstance(mirror_axes, str):
            if mirror_axes.lower() == 'none':
                mirror_axes = None
            else:
                mirror_axes = tuple(int(x) for x in mirror_axes.split(','))
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        # Debug
        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}, initial_patch_size: {initial_patch_size}, patch_size: {patch_size}, dim: {dim}, mirror_axes: {mirror_axes}')
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def _build_loss(self):
        loss = DC_SkelREC_Diff_ClusterSM_and_CE_TopK_loss(
            ce_class_weight=self.ce_class_weight,
            n_classes_w_bg=self.n_classes_w_bg,
            weight_ce=self.weight_ce,
            weight_topk=self.weight_topk,
            weight_dice=self.weight_dice,
            weight_srec=self.weight_srec,
            weight_diff=self.weight_diff,
            weight_lcluster=self.weight_lcluster,
            dice_batch_dice=self.dice_batch_dice,
            dice_do_bg=self.dice_do_bg
        ).to(self.device)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            n = len(deep_supervision_scales)                 # total decoder outputs (full-res first)
            nd = getattr(self, 'num_ds_levels', None)        # # of down-sampled levels to supervise
            k = (n - 1) if nd is None else max(0, min(int(nd), n - 1))   # last supervised level index

            # exponentially decaying weights (division by 2) so higher-res outputs dominate; levels
            # beyond `k` (i.e. more than `num_ds_levels` down-sampled maps) get zero weight.
            weights = np.array([1 / (2 ** i) if i <= k else 0 for i in range(n)], dtype=float)
            weights = weights / weights.sum()
            # full loss at full-res (level 0), CE + Dice only on the down-sampled levels
            loss = DeepSupervisionFullPlusCEDice(loss, weights)
        return loss
    
    def get_tr_dataset(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=self.num_images_properties_loading_threshold)
        return dataset_tr
    
    def build_train_dataloader(self, dataset_tr, initial_patch_size, tr_transforms, dim):
        """Construct the training dataloader object. Factored out of get_dataloaders so the choice of dataloader is
        driven by attributes (set in __init__/subclasses) without copying the rest of get_dataloaders.

        When self.cls_balanced_global_sampling is on (3D only), use the globally class-balanced foreground sampler
        with always-probabilistic fg/bg oversampling; otherwise use the default Skel dataloaders.

        When self.case_sampling_weight_groups is set, per-case "potential" weights (by name pattern) are turned
        into sampling_probabilities for the background/default case draw; the cls-balanced loader additionally
        applies the same weights to its within-class foreground case draw."""
        sampling_probabilities = self.build_case_sampling_probabilities(dataset_tr)
        if dim == 3 and self.cls_balanced_global_sampling:
            return nnUNetDataLoader3DSkelClsBalancedGlobal(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None,
                                       probabilistic_oversampling=True, transforms=tr_transforms,
                                       fg_labels=self.label_manager.foreground_labels,
                                       fg_class_sampling_weights=self.fg_class_sampling_weights,
                                       fg_class_calibration_degree=self.fg_class_calibration_degree,
                                       case_sampling_weight_groups=self.case_sampling_weight_groups,
                                       print_to_log_file=self.print_to_log_file)
        if dim == 2:
            dl_tr = nnUNetDataLoader2DSkel(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None, transforms=tr_transforms)
        else:
            dl_tr = nnUNetDataLoader3DSkel(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None, transforms=tr_transforms)
        return dl_tr

    def build_case_sampling_probabilities(self, dataset_tr):
        """Turn self.case_sampling_weight_groups into a per-case probability vector aligned to the dataloader's
        case order (list(dataset_tr.keys())), for use as sampling_probabilities in the background/default case draw.
        Returns self.sampling_probabilities unchanged when no weight groups are configured."""
        if not self.case_sampling_weight_groups:
            return self.sampling_probabilities
        keys = list(dataset_tr.keys())
        w = np.array([case_sampling_weight(k, self.case_sampling_weight_groups) for k in keys], dtype=np.float64)
        assert w.sum() > 0, "case_sampling_weight_groups zeroed out every case's weight"
        probs = w / w.sum()
        n_non_default = int(np.sum(w != 1.0))
        # the per-case list is long; write it to a sidecar file next to the training log and keep stdout short
        weights_file = join(os.path.dirname(self.log_file), 'case_sampling_weights.txt')
        with open(weights_file, 'w') as f:
            f.write(f"# case_sampling_weight_groups: {self.case_sampling_weight_groups}\n")
            f.write(f"# {n_non_default}/{len(keys)} cases have a non-default weight\n")
            f.write("# case_key\tweight\tprob\n")
            for k, wk, p in zip(keys, w, probs):
                f.write(f"{k}\t{wk:g}\t{p:.6g}\n")
        self.print_to_log_file(f"[case_sampling_weight_groups] {self.case_sampling_weight_groups}: "
                               f"{n_non_default}/{len(keys)} cases have a non-default weight; "
                               f"per-case list written to {weights_file}")
        return probs

    def get_dataloaders(self):
        """In fact, the val dataloader is None here."""
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr = self.get_tr_dataset()

        dl_tr = self.build_train_dataloader(dataset_tr, initial_patch_size, tr_transforms, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2) if self.num_cached is None else self.num_cached, seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        return mt_gen_train, None
    
    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=self.p_rotation,
                rotation=rotation_for_DA, p_scaling=self.p_scaling, scaling=self.scaling, p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        transforms.append(SkeletonTransform(do_tube=True))

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        skel = batch['skel']
        # print('type skel:', type(batch['skel']))
        # print('keys', batch['keys'])
        # print(list(batch.keys()))
        # for k in batch:
        #     if isinstance(batch[k], (torch.Tensor, np.ndarray)):
        #         print(f"{k}: {batch[k].shape}")
        #     elif isinstance(batch[k], str):
        #         print(f"{k}: {batch[k]}")
        #     elif isinstance(batch[k], list):
        #         print(f"{k}: {[x if isinstance(x, str) else x.shape if isinstance(x, (torch.Tensor, np.ndarray)) else 'N/A' for x in batch[k]]}")
        
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0].cpu().numpy(), name='data')
        # viewer.add_image(target[0][0].cpu().numpy(), name='target')
        # viewer.add_image(skel[0][0].cpu().numpy(), name='skel')
        # napari.run()
        

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        skel = skel.to(self.device, non_blocking=True)   # always a single full-res tensor

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target, skel)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def perform_actual_validation(self, save_probabilities: bool = False):
        st0 = time()
        assert self.configuration_manager.next_stage_names is None, f"Unsupported next_stage_names: {self.configuration_manager.next_stage_names}"
        assert not self.is_cascaded, f"Unsupported is_cascaded: {self.is_cascaded}"

        # During inference the decoder must return a single full-res tensor, not the deep-supervision
        # list of outputs (otherwise the predictor's [0] grabs the full-res list element, keeping its
        # batch dim, and predicted_logits[sl] += prediction broadcast-fails). Restored at the end.
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(self.segmentation_export_pool_size) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split(verbose=False)
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=self.num_images_properties_loading_threshold)

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                # self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                # self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else self.metrics_num_processes,
                                                compute_hd95=self.compute_hd95,
                                                to_compute_contamination=self.to_compute_contamination,
                                                to_compute_dwe=self.to_compute_dwe,
                                                dwe_max_dist=self.dwe_max_dist,
                                                dwe_norm_by=self.dwe_norm_by,
                                                dwe_edt_cache_dir=self.dwe_edt_cache_dir,
                                                dwe_weight_fn=self.dwe_weight_fn,
                                                dwe_weight_margin=self.dwe_weight_margin,
                                                to_compute_topbrain=self.to_compute_topbrain,
                                                topbrain_track=self.topbrain_track)
            _avg_dice = metrics['foreground_mean']["Dice"]
            _per_class_dice = [_rd(metrics['mean'][x]["Dice"]) for x in range(1, self.n_classes_w_bg)]
            self.print_to_log_file(f"[*] Val avg_dice: {_rd(_avg_dice)}, {time()-st0:.0f}s", also_print_to_console=True)
            self.print_to_log_file(f"[*] Val per_class_dice: {_per_class_dice}", also_print_to_console=True)
            self.logger.log('mean_fg_dice', _avg_dice, self.current_epoch)
            self.logger.log('dice_per_class_or_region', {x: metrics['mean'][x]["Dice"] for x in range(1, self.n_classes_w_bg)}, self.current_epoch)

        compute_gaussian.cache_clear()

        # restore deep supervision for the remaining training epochs
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

    def is_val_epoch(self, epoch):
        return (epoch+1) % self.num_epochs_per_val == 0
        # return epoch+1 == self.num_epochs or (epoch+1) % self.num_epochs_per_val == 0
        # return False
    
    def on_epoch_end(self):
        """If there is validation, it is done before `on_epoch_end`."""
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file(
            f"train_loss: {_rd(self.logger.my_fantastic_logging['train_losses'][-1])}"
            f", Epoch time: {_rd(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing
        if self.is_val_epoch(current_epoch):
            cur_score = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
            if self._best_ema is None or cur_score > self._best_ema:
                self._best_ema = cur_score
                self.print_to_log_file(f"===========> New best mean_fg_dice: {_rd(self._best_ema)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
    
    def run_training(self):
        self.on_train_start()

        # Lets run_training.py skip its post-training validation when the final epoch was already
        # validated in-loop. Initialized here (not __init__) because the EnvCfg subclass bypasses
        # this class's __init__.
        self.final_epoch_validated_in_loop = False

        # `self.current_epoch` starts from 0
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            if self.is_val_epoch(epoch):
                with torch.no_grad():
                    self.perform_actual_validation()
                if epoch + 1 == self.num_epochs:
                    self.final_epoch_validated_in_loop = True

            self.on_epoch_end()  # `self.current_epoch += 1` in this function

        self.on_train_end()
