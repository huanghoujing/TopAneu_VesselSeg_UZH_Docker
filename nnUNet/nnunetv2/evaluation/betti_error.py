from enum import Enum
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from skimage.measure import euler_number, label


"""
Betti error calculation taken from the TopCoW evaluation script
https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py
"""


def extract_labels(*, gt_array: np.array, pred_array: np.array) -> List:
    """Extracts union of labels in gt and pred masks"""
    labels_gt = np.unique(gt_array)
    labels_pred = np.unique(pred_array)
    labels = list(set().union(labels_gt, labels_pred))
    labels = [int(x) for x in labels]
    return labels


def filter_mask_by_label(mask: np.array, label: int) -> np.array:
    """
    filter the mask (numpy array), keep the voxels matching the label as 1
        convert the voxels that are not matching the label as 0
    """
    return np.where(mask == label, 1, 0)


def betti_number(img: np.array) -> List:
    """
    calculates the Betti number B0, B1, and B2 for a 3D img
    from the Euler characteristic number

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    """

    # make sure the image is 3D (for connectivity settings)
    assert img.ndim == 3

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1
    N26 = 3

    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)

    # make sure the image is binary with
    assert set(np.unique(padded)).issubset({0, 1})

    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1

    # get the label connected regions for foreground
    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    # get the label connected regions for background
    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    # NOTE: need to substract 1 from b2
    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    # print(f"Betti number: b0 = {b0}, b1 = {b1}, b2 = {b2}")

    return [b0, b1, b2]


def betti_number_error_all_classes(*, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict:
    """
    If task is TASK.BINARY_SEGMENTATION,
        it will compute the CoW class betti number error
    If task is TASK.MULTICLASS_SEGMENTATION, it will
        compute betti number errors of union of classes and an overall average per case.

    NOTE: returned betti_num_err_dict only considers all labels
        which are present in both gt and pred to compute the per-case-average
    """
    # print("\n-- call betti_number_error_all_classes()")

    labels = extract_labels(gt_array=gt_array, pred_array=pred_array)
    labels.remove(0)

    # when there are no labels in the ROI,
    # return blank betti_num_err_dict with only B0err_average and merged_binary of 0
    if len(labels) == 0:
        return {
            "Betti_0_error": 0,
            "Betti_1_error": 0
        }
    # otherwise compute the Betti 0 number error for label in union
    # and update the betti_num_err_dict

    sum_betti_0 = 0
    sum_betti_1 = 0

    for voxel_label in labels:
        # filter the view by that label
        filtered_gt = filter_mask_by_label(gt_array, voxel_label)
        filtered_pred = filter_mask_by_label(pred_array, voxel_label)

        gt_betti_numbers = betti_number(filtered_gt)
        pred_betti_numbers = betti_number(filtered_pred)

        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])

        sum_betti_0 += betti_0_error
        sum_betti_1 += betti_1_error

    overall_betti_0_error = sum_betti_0 / len(labels)
    overall_betti_1_error = sum_betti_1 / len(labels)

    return {
        "Betti_0_error": overall_betti_0_error,
        "Betti_1_error": overall_betti_1_error
    }