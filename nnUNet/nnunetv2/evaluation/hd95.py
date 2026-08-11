import numpy as np
import SimpleITK as sitk
from typing import Union


def compute_hd95(gt: Union[sitk.Image, np.ndarray], pred: Union[sitk.Image, np.ndarray], 
                 spacing: Union[None, np.ndarray]=None, ):
    """
    evaluation code as in Toothfairy challenge
    https://github.com/AImageLab-zip/ToothFairy/blob/main/evaluation/evaluation.py
    """

    # Edge case: if gt and pred are both empty, return 0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return np.nan 

    if isinstance(gt, np.ndarray):
        gt = sitk.GetImageFromArray(gt)
    if isinstance(pred, np.ndarray):
        pred = sitk.GetImageFromArray(pred)
    if spacing is not None:
        gt.SetSpacing(spacing.astype(np.float64))
        pred.SetSpacing(spacing.astype(np.float64))

    signed_distance_map = sitk.SignedMaurerDistanceMap(
        gt, squaredDistance=False, useImageSpacing=True
    )

    ref_distance_map = sitk.Abs(signed_distance_map)
    ref_surface = sitk.LabelContour(gt, fullyConnected=True)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())


    signed_distance_map_pred = sitk.SignedMaurerDistanceMap( pred, squaredDistance=False, useImageSpacing=True)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    seg_surface = sitk.LabelContour(pred > 0.5, fullyConnected=True)

    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)

    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(seg_surface > 0.5)

    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.percentile(all_surface_distances, 95)