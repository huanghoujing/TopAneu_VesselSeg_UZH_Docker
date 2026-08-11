import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_cbdice_loss import DC_and_CE_and_CBDC_loss


class nnUNetTrainerNoMirroringCBDice(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_srec = 1 # This is the default value, you can change it if you want

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def _build_loss(self):
        lambda_cbdice = 1.0
        lambda_dice = 1.0
        lambda_ce = 1.0

        loss = DC_and_CE_and_CBDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'iter_': 10, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_cbdice=lambda_cbdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
    
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        self.print_to_log_file("lambda_cbdice: %s" % str(lambda_cbdice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss
    