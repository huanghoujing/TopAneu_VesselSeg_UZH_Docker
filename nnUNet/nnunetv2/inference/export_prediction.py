import os
from copy import deepcopy
from typing import Union, List

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset, comp_blosc2_params

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping

def convert_predicted_logits_to_prob_with_correct_shape(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    num_threads_torch: int = default_num_processes
):
    """HOUJING:
    No argmax, only probabilities (numpy array) are returned.
    Also, device management for torch is added.
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    # predicted_logits = configuration_manager.resampling_fn_probabilities(
    predicted_logits = resample_torch_fornnunet(
        predicted_logits,
        properties_dict['shape_after_cropping_and_before_resampling'],
        current_spacing,
        properties_dict['spacing'],
        # device=predicted_logits.device if isinstance(predicted_logits, torch.Tensor) else torch.device('cpu')
        device=torch.device('cpu')
    )
    # TODO: TMP: nearest
    # assert len(predicted_logits.shape) == 4, f"Invalid shape {predicted_logits.shape}"
    # predicted_logits = torch.nn.functional.interpolate(
    #     predicted_logits.unsqueeze(0),
    #     size=properties_dict['shape_after_cropping_and_before_resampling'],
    #     mode='nearest',
    # )[0]
    if isinstance(predicted_logits, torch.Tensor):
        predicted_logits = predicted_logits.cpu()
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    
    # revert cropping
    predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities, properties_dict['bbox_used_for_cropping'], properties_dict['shape_before_cropping'])
    predicted_probabilities = predicted_probabilities.cpu().numpy()
    # revert transpose
    predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
    torch.set_num_threads(old_threads)
    return predicted_probabilities

def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)

def export_prediction_from_logits_to_cls_prob(
    predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
    save_probabilities: bool = False):
    """HOUJING:
    No resampling is done.
    Max prob per aneurysm channel are returned.
    """

    assert len(predicted_array_or_file.shape) == 4, f"Invalid shape {predicted_array_or_file.shape}"
    per_ch_probs = torch.amax(torch.softmax(predicted_array_or_file, dim=0), dim=(1,2,3))
    out_dir = os.path.dirname(output_file_truncated)
    torch.save(per_ch_probs, output_file_truncated + '_prob.pth')

def DEPRECATED_export_prediction_from_logits_no_resample(
    predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
    save_probabilities: bool = False):
    """HOUJING:
    No resampling is done.
    """
    assert len(predicted_array_or_file.shape) == 4, f"Invalid shape {predicted_array_or_file.shape}"

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)

    # save
    if save_probabilities:
        probabilities_final = torch.softmax(predicted_array_or_file, dim=0)
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
    
    segmentation_final = torch.argmax(predicted_array_or_file, dim=0).cpu().numpy().astype(np.uint8)

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)

def export_prediction_from_logits_no_resample_nibabel_dummy_affine(
    predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
    save_probabilities: bool = False):
    """HOUJING:
    No resampling is done.
    Nibabel and dummy affine are used.
    """
    assert len(predicted_array_or_file.shape) == 4, f"Invalid shape {predicted_array_or_file.shape}"

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)

    # save
    if save_probabilities:
        probabilities_final = torch.softmax(predicted_array_or_file, dim=0)
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
    
    segmentation_final = torch.argmax(predicted_array_or_file, dim=0).cpu().numpy().astype(np.uint8)  # shape [H,W,D]
    img_nib = nib.Nifti1Image(segmentation_final, np.eye(4))
    os.makedirs(os.path.dirname(output_file_truncated), exist_ok=True)
    nib.save(img_nib, f"{output_file_truncated}.nii.gz")
    

def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    # blosc2 hard switch: this is the cascade's predicted-next-stage seg, which is read back by the
    # dataset as a {c}.b2nd prev-stage file -> write .b2nd (not .npz) to keep the format contract.
    segmentation = segmentation.astype(np.uint8)
    block_size, chunk_size = comp_blosc2_params(
        (1, *segmentation.shape), tuple(configuration_manager.patch_size), segmentation.itemsize)
    nnUNetDataset.save_seg(segmentation, output_file,
                           chunks_seg=tuple(int(i) for i in chunk_size[1:]),
                           blocks_seg=tuple(int(i) for i in block_size[1:]))
    torch.set_num_threads(old_threads)
