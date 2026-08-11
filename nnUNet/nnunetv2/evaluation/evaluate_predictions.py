import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional
import inspect
import time

import SimpleITK as sitk
from SimpleITK import GetArrayViewFromImage as ArrayView
# from topbrain25_eval.utils.utils_mask import arr_is_binary, pad_sitk_image

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.evaluation.cldice import clDice
from nnunetv2.evaluation.hd95 import compute_hd95
from nnunetv2.evaluation.betti_error import betti_number_error_all_classes
from nnunetv2.evaluation.contamination_ratio_and_num_src import compute_contamination

def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def get_sitk_img_diag_len(img):
    # Load your image
    # img = sitk.ReadImage("image.nii.gz")

    # 1. Get voxel counts (discrete dimensions)
    size = img.GetSize()   # (W, H, D) in number of voxels

    # 2. Get voxel spacing (physical size per voxel)
    spacing = img.GetSpacing()  # (sx, sy, sz) in mm (or whatever unit)

    # 3. Compute physical dimensions along each axis
    physical_size = [s * sp for s, sp in zip(size, spacing)]
    # [width_mm, height_mm, depth_mm]

    # 4. Diagonal length (physical space)
    diagonal = np.linalg.norm(physical_size)

    # print("Voxel counts (W,H,D):", size)
    # print("Voxel spacing:", spacing)
    # print("Physical size (mm):", physical_size)
    # print("Physical diagonal (mm):", diagonal)
    return diagonal


# https://github.com/CoWBenchmark/TopBrain_Eval_Metrics/blob/master/topbrain25_eval/utils/utils_mask.py
def arr_is_binary(arr: np.array) -> bool:
    """
    test if the numpy array is binary
    NOTE: all zeros or all ones are also binary!
    """
    return set(np.unique(arr)).issubset({0, 1})

# https://github.com/CoWBenchmark/TopBrain_Eval_Metrics/blob/master/topbrain25_eval/utils/utils_mask.py
def pad_sitk_image(image: sitk.Image) -> sitk.Image:
    # print("\nbefore padding, image:\n")
    # print(image.GetSize())
    # print(sitk.GetArrayFromImage(image))

    # Define the amount of padding to add to each side (x, y, z)
    # https://itk.org/Doxygen/html/classitk_1_1PadImageFilter.html
    dim = image.GetDimension()  # can be 2D or 3D
    pad_lower_bound = [1] * dim  # Padding to add at the beginning of each axis
    pad_upper_bound = [1] * dim  # Padding to add at the end of each axis

    # Pad the image with 0s
    constant = 0
    padded_image = sitk.ConstantPad(image, pad_lower_bound, pad_upper_bound, constant)

    # print("\nafter padding, padded_image:\n")
    # print(padded_image.GetSize())
    # print(sitk.GetArrayFromImage(padded_image))

    return padded_image

# https://github.com/CoWBenchmark/TopBrain_Eval_Metrics/blob/master/topbrain25_eval/metrics/cls_avg_hd95.py
# HOUJING: If both gt and pred are empty for `label`, just use HD95 = HD = 0, and don't use this function.
def hd95_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int, HD95_UPPER_BOUND=290) -> list[float]:
    """
    Calculates the Hausdorff distance at 95% percentile

    NOTE: While there are many different implementations,
    packages, and even definitions(!) to calculate HD95,
    we decide to go with the definiton from
        Reinke, A., Tizabi, M.D., Baumgartner, M. et al.
        Understanding metric-related pitfalls in image analysis validation.
        Nat Methods 21, 182–194 (2024).
        See Fig. SN 3.63 and
        https://metrics-reloaded.dkfz.de/metric?id=hd95
    The implementation takes the max of two d_95:
        max(d_95(A,B), d_95(B,A))
    We verified this implementation with various Figs from
        Reinke et al., 2021
        Common Limitations of Image Processing Metrics: A Picture Story

    NOTE: in case of missing values (FP or FN), set the HD95
    to be roughly the maximum distance in ROI = 90 mm (HD95_UPPER_BOUND)

    Parameters
    ----------
    gt:
        ground truth mask sitk image
    pred:
        predicted mask sitk image
    label:
        annotation label integer

    Returns
    ----------
    [float hd95_score, float hd100_score]
    The distance unit is the same as the voxelspacing,
        which is usually in mm.

    References:
        Reinke et al., 2024
            Metrics reloaded: recommendations for image analysis validation
        Reinke et al., 2021
            Common Limitations of Image Processing Metrics: A Picture Story
        ITK forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832
        ITK tutorial surface_hausdorff_distance:
            NOTE: use the latest InsightSoftwareConsortium/SimpleITK-Notebooks repo
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
            with fixes from:
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/0cb643655d9fc6f08cecffc1ffe1d0997d78dedb
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py
    """
    # print(f"\n--> hd95_single_label(label-{label})\n")

    # HOUJING:
    #   For a label, if you want to treat the total-false-positive or total-false-negative in different samples the same,
    #       then use a fixed upper bound.
    #   If you want to treat samples differently, then `get_sitk_img_diag_len` can be specific. In this case, total-FP or
    #       total-FN mistakes in different samples are punished differently.
    if HD95_UPPER_BOUND is None:
        HD95_UPPER_BOUND = get_sitk_img_diag_len(gt)

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D (allow for 2D for testing purposes)
    assert gt.GetDimension() in (2, 3), "sitk img in 2D|3D, only fo HD"

    # NOTE: need to pad the image in case it is completely filled
    # found that when the image is completely filled,
    # SignedMaurerDistanceMap does not work as it gives all 0 distance.
    # see issue: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/issues/453
    # fix by pad the gt and pred

    gt = pad_sitk_image(gt)
    pred = pad_sitk_image(pred)

    # only need bool binary mask of the current label
    gt_label_img = gt == label
    pred_label_img = pred == label

    # gt_arr and pred_arr are from union of showed-up labels,
    # thus they will not be both all zeros
    # thus only FP and FN can happen
    # handle FP and FN with HD95_UPPER_BOUND

    gt_label_arr = sitk.GetArrayFromImage(gt_label_img)
    pred_label_arr = sitk.GetArrayFromImage(pred_label_img)

    # make sure the masks are binary
    assert arr_is_binary(gt_label_arr), "hd95_single_label expects binary gt_arr"
    assert arr_is_binary(pred_label_arr), "hd95_single_label expects binary pred_arr"

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        # print(f"[!!Warning] label-{label} empty for gt or pred")
        return [HD95_UPPER_BOUND, HD95_UPPER_BOUND]

    ##################################################################
    # Now the real HD95 implementation :)
    # -> max(d_95(A,B), d_95(B,A))
    ##################################################################

    # get the distance_map, surface, and number of surface pixels
    # for both gt/ref and pred
    (
        ref_distance_map,
        ref_surface,
        num_ref_surface_pixels,
    ) = _get_surface_distance(gt_label_img)
    (
        pred_distance_map,
        pred_surface,
        num_pred_surface_pixels,
    ) = _get_surface_distance(pred_label_img)

    # extract the distances of boundary_ref to boundary_pred
    # and vice versa for both directions
    # NOTE: SimpleITK MultiplyImageFilter requires
    # both input images to have the same pixel type
    # distance_map is float32, so need to cast surface to float
    ref2pred_distance_map = pred_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
    pred2ref_distance_map = ref_distance_map * sitk.Cast(pred_surface, sitk.sitkFloat32)

    # with np.printoptions(precision=1, suppress=True):
    #     print("ref2pred_distance_map =\n", ArrayView(ref2pred_distance_map))
    #     print("pred2ref_distance_map =\n", ArrayView(pred2ref_distance_map))

    # extract the non-zero distances from the distance_map
    ref2pred_distances = list(
        ArrayView(ref2pred_distance_map)[ArrayView(ref2pred_distance_map) != 0]
    )
    # create a list based on the number of surface pixels
    # populate the rest of the list with 0
    ref2pred_distances += [0] * (num_ref_surface_pixels - len(ref2pred_distances))

    # print("ref2pred_distances =\n", sorted(ref2pred_distances, reverse=True))
    # print("# ref2pred_distances =\n", len(ref2pred_distances))

    # do the same for ther other direction pred2ref
    pred2ref_distances = list(
        ArrayView(pred2ref_distance_map)[ArrayView(pred2ref_distance_map) != 0]
    )
    pred2ref_distances += [0] * (num_pred_surface_pixels - len(pred2ref_distances))

    # print("pred2ref_distances =\n", sorted(pred2ref_distances, reverse=True))
    # print("# pred2ref_distances =\n", len(pred2ref_distances))

    # use formula -> max(d_95(A,B), d_95(B,A))
    d_95_ref2pred = np.percentile(ref2pred_distances, 95)
    # print("d_95_ref2pred = ", d_95_ref2pred)
    d_95_pred2ref = np.percentile(pred2ref_distances, 95)
    # print("d_95_pred2ref = ", d_95_pred2ref)

    hd95_score = max(d_95_ref2pred, d_95_pred2ref)
    # print("hd95_score = ", hd95_score)

    # also keep track of HD max
    d_100_ref2pred = np.percentile(ref2pred_distances, 100)
    # print("d_100_ref2pred = ", d_100_ref2pred)
    d_100_pred2ref = np.percentile(pred2ref_distances, 100)
    # print("d_100_pred2ref = ", d_100_pred2ref)
    hd100_score = max(d_100_ref2pred, d_100_pred2ref)
    # print("hd100_score = ", hd100_score)

    return [hd95_score, hd100_score]


# https://github.com/CoWBenchmark/TopBrain_Eval_Metrics/blob/master/topbrain25_eval/metrics/cls_avg_hd95.py
def _get_surface_distance(seg: sitk.Image) -> tuple[sitk.Image, sitk.Image, int]:
    """
    Code adapted from:
        ITK Forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/
        ITK tutorial surface_hausdorff_distance:
            NOTE: use the latest InsightSoftwareConsortium/SimpleITK-Notebooks repo
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
            with fixes from:
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/0cb643655d9fc6f08cecffc1ffe1d0997d78dedb
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py

    NOTE: in bugfix https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
    "BUG: Used segmentation distance maps and not surface distance maps.
        Distances between surfaces should use the surface distance maps and
        not distance maps based on the original segmentations."
    """

    # extract the contour outline for later masking
    seg_surface = sitk.LabelContour(
        seg,
        # set to fully connected
        fullyConnected=True,
    )

    # get map of the distance to boundary for input segmentation mask
    # use image spacing with Maurer distance transform
    seg_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            seg_surface,  # fix in SimpleITK-Notebooks/commit/4a3967
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    # with np.printoptions(precision=1, suppress=True):
    #     print("seg_distance_map =\n", ArrayView(seg_distance_map))
    #     print("seg_distance_map.GetSize() =", seg_distance_map.GetSize())

    #     print("seg_surface =\n", ArrayView(seg_surface))
    #     print("seg_surface.GetSize() =", seg_surface.GetSize())

    # get the number of surface pixels for HD sorting later
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(seg_surface)

    num_surface_pixels = int(statistics_image_filter.GetSum())
    # print("num_surface_pixels = ", num_surface_pixels)

    return seg_distance_map, seg_surface, num_surface_pixels



def keep_labels_1to34_except_4_6(img: sitk.Image) -> sitk.Image:
    """Example gt_hook/pred_hook: keep label value x iff (1 <= x <= 34) and x not in (4, 6); set all other
    voxels to 0. Module-level (hence picklable) so it can be shipped to spawn worker processes."""
    arr = sitk.GetArrayFromImage(img)
    keep = (arr >= 1) & (arr <= 34) & (arr != 4) & (arr != 6)
    out = sitk.GetImageFromArray(np.where(keep, arr, 0).astype(arr.dtype))
    out.CopyInformation(img)
    return out


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None, compute_hd95=False, HD95_UPPER_BOUND=290,
                    nan_as_one=False,
                    to_compute_contamination=False,
                    to_compute_dwe=False, dwe_max_dist=500.0, dwe_norm_by=1e6, dwe_edt_cache_dir=None,
                    dwe_weight_fn='linear', dwe_weight_margin=3.0,
                    gt_hook=None, pred_hook=None) -> dict:
    """
    gt_hook / pred_hook:
        Optional callables `sitk.Image -> sitk.Image` applied to the gt/pred image right after reading and
        BEFORE it is turned into an array, so the processed version feeds BOTH the array-based metrics
        (Dice/IoU) and the sitk-based metrics (HD95/contamination) consistently. Use them e.g. to erase some
        labels or remap label values. Must be picklable (module-level functions, not lambdas) because they
        are shipped to worker processes via spawn multiprocessing.

        Example (keep label value x iff 1 <= x <= 34 and x != 4 and x != 6, set all others to 0) -- see the
        module-level `keep_labels_1to34_except_4_6`:

            def keep_labels_1to34_except_4_6(img: sitk.Image) -> sitk.Image:
                arr = sitk.GetArrayFromImage(img)
                keep = (arr >= 1) & (arr <= 34) & (arr != 4) & (arr != 6)
                out = sitk.GetImageFromArray(np.where(keep, arr, 0).astype(arr.dtype))
                out.CopyInformation(img)
                return out

            compute_metrics(ref, pred, ..., gt_hook=keep_labels_1to34_except_4_6,
                            pred_hook=keep_labels_1to34_except_4_6)
    Shape change:
        SimpleITKIO.read_seg returns a 4D array (1, Z, Y, X);
        GetArrayFromImage returns 3D (Z, Y, X). This is harmless here because everything downstream 
        (region_or_label_to_mask, compute_tp_fp_fn_tn) is element-wise, and ref/pred are now both 
        3D and aligned. The HD95/contamination paths already used the sitk images, unaffected.
    Reader abstraction:
        compute_metrics is now hardwired to SimpleITK-readable files (.nii.gz/.mha), rather than
        going through the passed-in image_reader_writer. That's fine for this TopCoW pipeline, 
        and the HD95/contamination branches already assumed it. I left image_reader_writer in 
        the signature since compute_metrics_on_filelist passes it positionally via starmap.
    """
    # load images: read once with SimpleITK and derive the label arrays from the sitk images, so that the
    # Dice/IoU path (needs arrays) and the HD95/contamination path (needs sitk images) share a single read.
    # NOTE: image_reader_writer is kept in the signature for call compatibility but is no longer used here.
    gt_sitk = sitk.ReadImage(reference_file)
    pred_sitk = sitk.ReadImage(prediction_file)
    # user hooks operate on the sitk image, before it becomes an array
    if gt_hook is not None:
        gt_sitk = gt_hook(gt_sitk)
    if pred_hook is not None:
        pred_sitk = pred_hook(pred_sitk)
    seg_ref = sitk.GetArrayFromImage(gt_sitk)
    seg_pred = sitk.GetArrayFromImage(pred_sitk)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = 1 if nan_as_one else np.nan
            results['metrics'][r]['IoU'] = 1 if nan_as_one else np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
        if compute_hd95:
            if tp + fp + fn == 0:
                hd95_score, hd100_score = 0, 0
            else:
                hd95_score, hd100_score = hd95_single_label(gt=gt_sitk, pred=pred_sitk, label=r, HD95_UPPER_BOUND=HD95_UPPER_BOUND)
            results['metrics'][r]['HD95'] = hd95_score
            results['metrics'][r]['HD100'] = hd100_score

    if to_compute_contamination:
        contamination_metrics = compute_contamination(
            gt=gt_sitk,
            pred=pred_sitk,
            fg_fg_interface_margin=int(os.environ.get('CONTAMINATION_FG_FG_INTERFACE_MARGIN', 3)),
            bg_surface_margin=int(os.environ.get('CONTAMINATION_BG_SURFACE_MARGIN', 5)),
            UnderSeg_ratio_thresh=float(os.environ.get('CONTAMINATION_UNDERSEG_RATIO_THRESH', 0)),
            FGC_ratio_thresh=float(os.environ.get('CONTAMINATION_FGC_RATIO_THRESH', 0)),
            return_masks=False
        )
        results['metrics'].update(contamination_metrics)

    if to_compute_dwe:
        from nnunetv2.evaluation.distance_weighted_error import compute_distance_weighted_error
        # workers stay CPU (allow_gpu=False): cache reads + scipy fallback only. The GPU EDTs are
        # populated once by the single-process pre-pass in compute_metrics_on_filelist.
        dwe_metrics = compute_distance_weighted_error(
            gt=seg_ref, pred=seg_pred,
            weight_fn=dwe_weight_fn, weight_margin=dwe_weight_margin,
            max_dist=dwe_max_dist, dwe_norm_by=dwe_norm_by,
            edt_cache_dir=dwe_edt_cache_dir, gt_key=reference_file,
            allow_gpu=False,
        )
        results['metrics'].update(dwe_metrics)
    return results

def convert_nonnan_to_int(value):
    if np.isnan(value):
        return value
    else:
        return int(value)

def _env_float(name):
    """Read an env var as float, or None if unset/empty. Used to pin confusion-matrix PNG color
    scales (vmin/vmax) to a constant so heatmaps from different models are visually comparable."""
    v = os.environ.get(name, None)
    return float(v) if v not in (None, '') else None

def save_confusion_matrix_as_csv(confusion_matrix: np.ndarray, classes: List[int], output_file: str, verbose: bool = False, float_format: str = "%.0f"):
    import pandas as pd
    df_data = []
    for i, r in enumerate(classes):
        row = {'GT Label': r}
        for j, c in enumerate(classes):
            row['Pred '+str(c)] = confusion_matrix[i][j]
        df_data.append(row)
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, index=False, float_format=float_format)
    if verbose:
        print(f"[{inspect.currentframe().f_code.co_name}] Saved {output_file}")

def np_nan_aggregate_w_warning(in_list, func=np.nanmean, name=""):
    """
    Calculates the mean/std etc of a list while ignoring NaN values.
    Print a warning if NaN values are present in the input list.
    If all values are NaN, the result will be NaN.
    """
    num_nan = np.isnan(in_list).sum()
    if num_nan > 0:
        print(f"Warning [{name}]: {num_nan} NaN values found in the input list. They will be ignored in the mean calculation.")
    if num_nan == len(in_list):
        print(f"Warning [{name}]: All values are NaN. The result will be NaN.")
        return np.nan
    return func(in_list)


def _detect_topbrain_track(basename):
    """Detect 'ct'/'mr' from a case filename (e.g. topcow_ct_095.nii.gz, CT2MR_TnR_topcow_mr_001.nii.gz)."""
    low = basename.lower()
    if '_ct_' in low or low.startswith('ct') or 'topcow_ct' in low:
        return 'ct'
    if '_mr_' in low or low.startswith('mr') or 'topcow_mr' in low:
        return 'mr'
    raise ValueError(f"cannot detect ct/mr track from filename: {basename}")


_PARALLEL_TOPBRAIN_CLS = None


def _get_parallel_topbrain_cls():
    """Build (once) and return the _ParallelTopBrainEvaluation class.

    The class subclasses the bundled topbrain25_eval TopBrainEvaluation (nnunetv2/TopBrain_Eval_Metrics,
    added to sys.path lazily so importing this module never hard-depends on it). It must NOT be defined
    inside _run_topbrain_one_track: its score() submits the bound method self.score_case to a
    ProcessPoolExecutor, which pickles `self` -- and a class living in a function's local scope is not
    picklable ("Can't get local object ...<locals>._ParallelTopBrainEvaluation"). So we register it at
    module scope with a clean __qualname__/__module__; combined with a 'fork' pool context (workers
    inherit this registered global) the instance round-trips through pickle. Wrapper copied from
    compute_seg_metrics/eval_topbrain.py.
    """
    global _PARALLEL_TOPBRAIN_CLS
    if _PARALLEL_TOPBRAIN_CLS is not None:
        return _PARALLEL_TOPBRAIN_CLS
    import sys
    tb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TopBrain_Eval_Metrics')
    if tb_dir not in sys.path:
        sys.path.insert(0, tb_dir)
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from pandas import DataFrame
    from topbrain25_eval.evaluation import TopBrainEvaluation
    from topbrain25_eval.aggregate.aggregate_all_detection_dicts import aggregate_all_detection_dicts

    class _ParallelTopBrainEvaluation(TopBrainEvaluation):
        def __init__(self, *a, num_workers=1, **k):
            super().__init__(*a, **k)
            self.num_workers = num_workers

        def score(self):
            if self.num_workers <= 1:
                return super().score()
            # fork so the workers inherit this dynamically-registered class (and sys.path entry)
            with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=get_context('fork')) as ex:
                futs = [ex.submit(self.score_case, idx=idx, case=case)
                        for idx, case in self._cases.iterrows()]
                results = [f.result() for f in futs]
            self._case_results = DataFrame(results)
            self._aggregate_results = self.score_aggregates()
            # metric-6 Average F1 score, post-aggregated from the per-case detection dicts
            self._aggregate_results["dect_avg"] = aggregate_all_detection_dicts(
                self.track, self._case_results["all_detection_dicts"])

    # register at module scope so pickle can resolve instances by qualified name (see docstring)
    _ParallelTopBrainEvaluation.__module__ = __name__
    _ParallelTopBrainEvaluation.__qualname__ = '_ParallelTopBrainEvaluation'
    globals()['_ParallelTopBrainEvaluation'] = _ParallelTopBrainEvaluation
    _PARALLEL_TOPBRAIN_CLS = _ParallelTopBrainEvaluation
    return _ParallelTopBrainEvaluation


def _run_topbrain_one_track(track_str, ref_files, pred_files, work_dir, num_workers):
    """Symlink the given gt/pred files into temp folders, run the bundled TopBrain evaluator for one
    track, and return (n_cases, {'clsAvgDice', 'B0', 'F1'}) of aggregate means.

    Uses the topbrain25_eval package bundled in this repo (nnunetv2/TopBrain_Eval_Metrics), so it
    does not depend on the external compute_seg_metrics repo.
    """
    import sys
    import json
    import contextlib
    tb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TopBrain_Eval_Metrics')
    if tb_dir not in sys.path:
        sys.path.insert(0, tb_dir)
    from topbrain25_eval.constants import TRACK

    _ParallelTopBrainEvaluation = _get_parallel_topbrain_cls()

    gt_dir = os.path.join(work_dir, 'gt')
    pred_dir = os.path.join(work_dir, 'pred')
    out_dir = os.path.join(work_dir, 'out')
    for d in (gt_dir, pred_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    for src in ref_files:
        os.symlink(os.path.abspath(src), os.path.join(gt_dir, os.path.basename(src)))
    for src in pred_files:
        os.symlink(os.path.abspath(src), os.path.join(pred_dir, os.path.basename(src)))

    track = TRACK.CT if track_str == 'ct' else TRACK.MR
    # per-case scoring is very chatty; suppress its stdout/stderr like eval_topbrain_script.py does
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        _ParallelTopBrainEvaluation(
            track=track, expected_num_cases=len(pred_files),
            predictions_path=pred_dir, ground_truth_path=gt_dir, output_path=out_dir,
            num_workers=num_workers,
        ).evaluate()
    agg = json.load(open(os.path.join(out_dir, 'metrics.json')))['aggregates']
    return len(pred_files), {
        'clsAvgDice': agg['Dice_ClsAvgDice']['mean'],
        'B0':         agg['B0err_ClsAvgB0err']['mean'],
        'F1':         agg['dect_avg']['f1_score']['mean'],
    }


def _run_topbrain(files_ref, files_pred, topbrain_track, num_workers):
    """Run the TopBrain folder evaluation off the matched file lists and return a combined
    {'clsAvgDice', 'B0', 'F1'} dict (case-weighted mean across whatever tracks are present).

    topbrain_track: 'ct' / 'mr' (all files are that track) or 'auto' (split by filename). CT and MR
    have different label sets and side-road labels, so a mixed folder must be evaluated per track.
    """
    import tempfile
    import shutil
    pairs = list(zip(files_ref, files_pred))
    if topbrain_track in ('ct', 'mr'):
        groups = {topbrain_track: pairs}
    elif topbrain_track == 'auto':
        groups = {}
        for r, p in pairs:
            groups.setdefault(_detect_topbrain_track(os.path.basename(p)), []).append((r, p))
    else:
        raise ValueError(f"topbrain_track must be 'ct'/'mr'/'auto', got {topbrain_track}")

    work = tempfile.mkdtemp(prefix='topbrain_eval_')
    try:
        per_track = {}
        for tk, gp in groups.items():
            refs = [r for r, _ in gp]
            preds = [p for _, p in gp]
            n, m = _run_topbrain_one_track(tk, refs, preds, os.path.join(work, tk), num_workers)
            per_track[tk] = (n, m)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    total = sum(n for n, _ in per_track.values())
    # case-weighted mean across tracks; for F1 this is an approximation (F1 is aggregated from
    # per-case TP/FP/FN, not a per-case mean), exact when a single track is present.
    return {k: sum(n * m[k] for n, m in per_track.values()) / total
            for k in ('clsAvgDice', 'B0', 'F1')}


def build_one_line_score_columns(result, model_alias='', select=None, with_per_class=None, verbose=True):
    """Build the (header, row) lists for the one-line score CSV from a computed/loaded `result` dict.

    This is the single source of truth for the one-line-CSV columns, shared by
    `compute_metrics_on_filelist` (live) and the regen-from-summary.json script (offline) so a CSV can be
    re-emitted with a different column selection WITHOUT re-running evaluation -- everything below is read
    from `result`, which is exactly what gets saved to summary.json (the only extra input is `model_alias`).

    select: comma-separated header names (e.g. "Model,clsAvgDice,B0,F1,Dice"); when given, exactly those
        columns in that order, a requested-but-not-computed column written empty (stable schema). None -> the
        full auto pool below.
    with_per_class: '1' -> append per-class Dice_<label> columns; '0' -> never; None -> default (ON for the
        auto path, OFF for the explicit-select path).
    """
    # Assemble every available column as an ordered (header -> value) pool, with per-column rounding.
    # Contamination columns are explicitly selected and ordered via `con_cols` below.
    # fg_avg_con_* are totals;
    #   UnderSeg* = under-segmentation, FGC* = foreground confusion;
    #   BGC* = background contamination;
    # `_T` suffix = thresholded (only matrix entries with ratio > 0.05)
    # ConMat*/ConMatFgFg* = dataset-level contamination-matrix sums (each cell = mean ratio over
    # cases where the GT row-class is present, then summed; not diluted by per-case class count).
    avail = [('Model', model_alias),
             ('Dice', round(float(result['foreground_mean']['Dice']), 4))]
    if 'HD95' in result['foreground_mean']:                 # present iff compute_hd95
        avail.append(('HD95', round(float(result['foreground_mean']['HD95']), 2)))
    if 'overall_contamination' in result:                   # present iff to_compute_contamination
        oc = result['overall_contamination']
        con_cols = [
            # (overall_contamination key,            CSV header,        ndigits)
            ('fg_avg_con_ratio',                      'fg_avg_con_ratio', 5),
            ('fg_avg_con_sources',                    'fg_avg_con_sources', 3),
            ('UnderSeg_ratio',                        'UnderSeg_ratio',  5),
            ('UnderSeg_prevalence',                   'UnderSeg_prevalence', 3),
            ('FGC_ratio',                             'FGC_ratio',       5),
            ('FGC_sources',                           'FGC_sources',     3),
            ('UnderSeg_ratio_after_thresh',           'UnderSeg_ratio_T', 5),
            ('UnderSeg_prevalence_after_thresh',      'UnderSeg_prevalence_T', 3),
            ('FGC_ratio_after_thresh',                'FGC_ratio_T',     5),
            ('FGC_sources_after_thresh',              'FGC_sources_T',   3),
            # Dataset-level contamination-matrix sums (per-cell mean over cases, then summed; NOT
            # diluted by per-case class count). ConMat* = total over all cells, ConMatFgFg* = fg-fg
            # confusion columns only; `_T` = after-threshold matrix.
            ('con_cm_sum',                            'ConMatSum',       5),
            ('con_cm_fgfg',                           'ConMatFgFg',      5),
            ('con_cm_after_thresh_sum',               'ConMatSum_T',     5),
            ('con_cm_after_thresh_fgfg',              'ConMatFgFg_T',    5),
            ('BGC_voxels',                            'BGC_voxels',      1),
            ('BGC_sources',                           'BGC_sources',     3),
        ]
        for oc_key, csv_header, nd in con_cols:
            avail.append((csv_header, round(float(oc[oc_key]['mean']), nd)))   # each = mean over cases

    if 'overall_dwe' in result:                             # present iff to_compute_dwe
        od = result['overall_dwe']
        dwe_cols = [
            # (overall_dwe key,     CSV header,    ndigits)
            ('dwe_under_seg',       'dweUndSeg',   4),    # distance-weighted under-seg severity
            ('dwe_epp',             'dweEpp',      4),    # distance-weighted error-predicted-positive
            ('dwe_error',           'dweError',    4),    # total error mass (= UndSeg + Epp; ssDWE)
            ('dwe_epp_from_bg',     'dweEppBg',    4),    # EPP from background (= matrix Row0)
            ('dwe_epp_fg_conf',     'dweEppFg',    4),    # EPP FG<->FG confusion (= matrix FgFg)
            ('dwe_ghost_nvox',      'dweGhostVox', 1),    # total GT-absent ("ghost") voxels
            ('dwe_ghost_ncls',      'dweGhostCls', 2),    # number of ghost classes
            ('dwe_cm_sum',          'dweCMSum',    4),    # confusion-matrix total (= Row0+Col0+FgFg)
            ('dwe_cm_row0',         'dweCMRow0',   4),    # BG->FG mass (false positives, incl ghost)
            ('dwe_cm_col0',         'dweCMCol0',   4),    # FG->BG mass (under-segmentation)
            ('dwe_cm_fgfg',         'dweCMFgFg',   4),    # FG<->FG confusion mass
        ]
        for od_key, csv_header, nd in dwe_cols:
            avail.append((csv_header, round(float(od[od_key]['mean']), nd)))   # each = mean over cases

    if 'topbrain' in result:                                # present iff to_compute_topbrain
        for key in ('clsAvgDice', 'B0', 'F1'):
            avail.append((key, round(float(result['topbrain'][key]), 4)))

    # Per-class Dice columns (one per scored label/region). Each value is `result['mean'][r]['Dice']`
    # (mean over cases for that class), some of which may be NaN. Always added to the selectable pool.
    per_class = []
    for r in result['mean'].keys():
        if r == 0 or r == '0':                              # skip background, like foreground_mean
            continue
        per_class.append((f'Dice_{r}', round(float(result['mean'][r]['Dice']), 4)))

    # Explicit, ordered column selection via `select` (comma-separated header names). When set, exactly
    # those columns are written in that order; a requested column that was not computed this run is
    # written empty (so the CSV schema stays stable across appended rows). When unset, the default is the
    # full pool above. In BOTH paths the per-class Dice columns are appended (in pool order) iff
    # with_per_class == '1' -- default ON for the auto path (legacy behavior), default OFF for the
    # explicit-select path (so an explicit schema stays exact unless per-class is opted in).
    if isinstance(select, str) and select:
        avail_map = dict(avail)
        avail_map.update(per_class)
        header, row = [], []
        for name in (s.strip() for s in select.split(',') if s.strip()):
            header.append(name)
            if name in avail_map:
                row.append(avail_map[name])
            else:
                row.append('')
                if verbose:
                    print(f"[build_one_line_score_columns] one-line CSV: requested column "
                          f"'{name}' was not computed this run; writing it empty. "
                          f"Available: {', '.join(h for h, _ in avail) + ', ' + ', '.join(h for h, _ in per_class) if per_class else ', '.join(h for h, _ in avail)}")
        if with_per_class == '1':                          # opt-in: append per-class after the selection
            header += [h for h, _ in per_class]
            row += [v for _, v in per_class]
    else:
        cols = list(avail)
        if (with_per_class if with_per_class is not None else '1') == '1':
            cols += per_class
        header = [h for h, _ in cols]
        row = [v for _, v in cols]
    return header, row


def append_one_line_score_csv(result, csv_path, model_alias='', select=None, with_per_class=None,
                              verbose=True):
    """Append one score row to `csv_path` (writing the header first iff the file is empty/absent).
    Columns are built by build_one_line_score_columns(); see it for `select`/`with_per_class`."""
    import csv as _csv
    header, row = build_one_line_score_columns(result, model_alias=model_alias, select=select,
                                               with_per_class=with_per_class, verbose=verbose)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    # "no header" == empty/absent file -> create header first, then append the row
    write_header = (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as f:
        writer = _csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True,
                              compute_hd95=False, HD95_UPPER_BOUND=290,
                              nan_as_one=False,
                              to_compute_contamination=False,
                              to_compute_dwe=False, dwe_max_dist=500.0, dwe_norm_by=1e6,
                              dwe_edt_cache_dir=None, dwe_weight_fn='linear', dwe_weight_margin=3.0,
                              to_compute_topbrain=False, topbrain_track='auto',
                              gt_hook=None, pred_hook=None,
                              ) -> dict:
    """
    output_file must end with .json; can be None
    """
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    # DWE EDT cache defaults to a hidden dir next to the GT folder (GT is fixed -> reused across runs)
    if to_compute_dwe and dwe_edt_cache_dir is None:
        dwe_edt_cache_dir = join(folder_ref, '.dwe_edt_cache')

    # HOUJING: remove duplicate code by calling compute_metrics_on_filelist
    return compute_metrics_on_filelist(
        files_ref=files_ref,
        files_pred=files_pred,
        output_file=output_file,
        image_reader_writer=image_reader_writer,
        regions_or_labels=regions_or_labels,
        ignore_label=ignore_label,
        num_processes=num_processes,
        compute_hd95=compute_hd95, HD95_UPPER_BOUND=HD95_UPPER_BOUND,
        nan_as_one=nan_as_one,
        to_compute_contamination=to_compute_contamination,
        to_compute_dwe=to_compute_dwe, dwe_max_dist=dwe_max_dist, dwe_norm_by=dwe_norm_by,
        dwe_edt_cache_dir=dwe_edt_cache_dir,
        dwe_weight_fn=dwe_weight_fn, dwe_weight_margin=dwe_weight_margin,
        to_compute_topbrain=to_compute_topbrain, topbrain_track=topbrain_track,
        gt_hook=gt_hook, pred_hook=pred_hook,
    )

def compute_metrics_on_filelist(
    files_ref: List,
    files_pred: List,
    output_file: str,
    image_reader_writer: BaseReaderWriter,
    regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
    ignore_label: int = None,
    num_processes: int = default_num_processes,
    compute_hd95=False, HD95_UPPER_BOUND=290,
    nan_as_one=False,
    to_compute_contamination=False,
    to_compute_dwe=False, dwe_max_dist=500.0, dwe_norm_by=1e6, dwe_edt_cache_dir=None,
    dwe_weight_fn='linear', dwe_weight_margin=3.0,
    to_compute_topbrain=False, topbrain_track='auto',
    gt_hook=None, pred_hook=None,
    verbose: bool = True,
    ) -> dict:
    """
    output_file must end with .json; can be None
    gt_hook / pred_hook: optional picklable `sitk.Image -> sitk.Image` callables applied per case before
        metric computation (see compute_metrics).
    """
    start_time = time.time()
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
        out_dir = os.path.dirname(output_file)
        os.makedirs(out_dir, exist_ok=True)

    assert len(files_ref) == len(files_pred), f"{len(files_ref)} != {len(files_pred)}"

    # DWE: single-process GPU-primary pre-pass to populate the per-class EDT cache BEFORE the worker
    # pool, so workers only read the cache (no CUDA inside forked/spawned workers). Skipped if no
    # cache dir (workers then fall back to scipy CPU per case), or if the weighting is distance-free
    # (weight_fn='ones'), in which case no EDT is needed at all.
    if to_compute_dwe and dwe_edt_cache_dir is not None:
        from nnunetv2.evaluation.distance_weighted_error import (
            precompute_dwe_cache_for_files, weight_fn_is_distance_free)
        if weight_fn_is_distance_free(dwe_weight_fn):
            if verbose:
                print(f"[{inspect.currentframe().f_code.co_name}] DWE weight_fn='{dwe_weight_fn}' is "
                      f"distance-free; skipping the EDT pre-pass/cache entirely.")
        else:
            precompute_dwe_cache_for_files(files_ref, dwe_edt_cache_dir, allow_gpu=True, verbose=verbose)

    if verbose:
        print(f"[{inspect.currentframe().f_code.co_name}] Starting metric computation on {len(files_ref)} file pairs "
              f"using {num_processes} processes...")
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred), [compute_hd95] * len(files_pred), [HD95_UPPER_BOUND] * len(files_pred),
                     [nan_as_one] * len(files_pred),
                     [to_compute_contamination] * len(files_pred),
                     [to_compute_dwe] * len(files_pred), [dwe_max_dist] * len(files_pred),
                     [dwe_norm_by] * len(files_pred), [dwe_edt_cache_dir] * len(files_pred),
                     [dwe_weight_fn] * len(files_pred), [dwe_weight_margin] * len(files_pred),
                     [gt_hook] * len(files_pred), [pred_hook] * len(files_pred),
            ))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}

    if to_compute_contamination:
        # dataset level contamination metrics
        result['overall_contamination'] = {}
        # Bookkeep the config that produced these metrics (mirrors the compute_contamination call site
        # in compute_metrics_on_filelist; same env vars and defaults) so the summary is self-describing.
        result['overall_contamination']['contamination_config'] = results[0]['metrics']['contamination_config']
        result['overall_contamination']['fg_avg_con_ratio'] = {}
        result['overall_contamination']['fg_avg_con_ratio']['mean'] = np_nan_aggregate_w_warning([r['metrics']['fg_avg_con_ratio'] for r in results], func=np.nanmean, name="fg_avg_con_ratio")
        result['overall_contamination']['fg_avg_con_ratio']['std'] = np_nan_aggregate_w_warning([r['metrics']['fg_avg_con_ratio'] for r in results], func=np.nanstd, name="fg_avg_con_ratio")
        result['overall_contamination']['fg_avg_con_sources'] = {}
        result['overall_contamination']['fg_avg_con_sources']['mean'] = np_nan_aggregate_w_warning([r['metrics']['fg_avg_con_sources'] for r in results], func=np.nanmean, name="fg_avg_con_sources")
        result['overall_contamination']['fg_avg_con_sources']['std'] = np_nan_aggregate_w_warning([r['metrics']['fg_avg_con_sources'] for r in results], func=np.nanstd, name="fg_avg_con_sources")
        result['overall_contamination']['UnderSeg_ratio'] = {}
        result['overall_contamination']['UnderSeg_ratio']['mean'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_ratio'] for r in results], func=np.nanmean, name="UnderSeg_ratio")
        result['overall_contamination']['UnderSeg_ratio']['std'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_ratio'] for r in results], func=np.nanstd, name="UnderSeg_ratio")
        result['overall_contamination']['UnderSeg_prevalence'] = {}
        result['overall_contamination']['UnderSeg_prevalence']['mean'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_prevalence'] for r in results], func=np.nanmean, name="UnderSeg_prevalence")
        result['overall_contamination']['UnderSeg_prevalence']['std'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_prevalence'] for r in results], func=np.nanstd, name="UnderSeg_prevalence")
        result['overall_contamination']['FGC_ratio'] = {}
        result['overall_contamination']['FGC_ratio']['mean'] = np_nan_aggregate_w_warning([r['metrics']['FGC_ratio'] for r in results], func=np.nanmean, name="FGC_ratio")
        result['overall_contamination']['FGC_ratio']['std'] = np_nan_aggregate_w_warning([r['metrics']['FGC_ratio'] for r in results], func=np.nanstd, name="FGC_ratio")
        result['overall_contamination']['FGC_sources'] = {}
        result['overall_contamination']['FGC_sources']['mean'] = np_nan_aggregate_w_warning([r['metrics']['FGC_sources'] for r in results], func=np.nanmean, name="FGC_sources")
        result['overall_contamination']['FGC_sources']['std'] = np_nan_aggregate_w_warning([r['metrics']['FGC_sources'] for r in results], func=np.nanstd, name="FGC_sources")
        result['overall_contamination']['UnderSeg_ratio_after_thresh'] = {}
        result['overall_contamination']['UnderSeg_ratio_after_thresh']['mean'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_ratio_after_thresh'] for r in results], func=np.nanmean, name="UnderSeg_ratio_after_thresh")
        result['overall_contamination']['UnderSeg_ratio_after_thresh']['std'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_ratio_after_thresh'] for r in results], func=np.nanstd, name="UnderSeg_ratio_after_thresh")
        result['overall_contamination']['UnderSeg_prevalence_after_thresh'] = {}
        result['overall_contamination']['UnderSeg_prevalence_after_thresh']['mean'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_prevalence_after_thresh'] for r in results], func=np.nanmean, name="UnderSeg_prevalence_after_thresh")
        result['overall_contamination']['UnderSeg_prevalence_after_thresh']['std'] = np_nan_aggregate_w_warning([r['metrics']['UnderSeg_prevalence_after_thresh'] for r in results], func=np.nanstd, name="UnderSeg_prevalence_after_thresh")
        result['overall_contamination']['FGC_ratio_after_thresh'] = {}
        result['overall_contamination']['FGC_ratio_after_thresh']['mean'] = np_nan_aggregate_w_warning([r['metrics']['FGC_ratio_after_thresh'] for r in results], func=np.nanmean, name="FGC_ratio_after_thresh")
        result['overall_contamination']['FGC_ratio_after_thresh']['std'] = np_nan_aggregate_w_warning([r['metrics']['FGC_ratio_after_thresh'] for r in results], func=np.nanstd, name="FGC_ratio_after_thresh")
        result['overall_contamination']['FGC_sources_after_thresh'] = {}
        result['overall_contamination']['FGC_sources_after_thresh']['mean'] = np_nan_aggregate_w_warning([r['metrics']['FGC_sources_after_thresh'] for r in results], func=np.nanmean, name="FGC_sources_after_thresh")
        result['overall_contamination']['FGC_sources_after_thresh']['std'] = np_nan_aggregate_w_warning([r['metrics']['FGC_sources_after_thresh'] for r in results], func=np.nanstd, name="FGC_sources_after_thresh")
        result['overall_contamination']['BGC_voxels'] = {}
        result['overall_contamination']['BGC_voxels']['mean'] = np_nan_aggregate_w_warning([r['metrics']['BGC_voxels'] for r in results], func=np.nanmean, name="BGC_voxels")
        result['overall_contamination']['BGC_voxels']['std'] = np_nan_aggregate_w_warning([r['metrics']['BGC_voxels'] for r in results], func=np.nanstd, name="BGC_voxels")
        result['overall_contamination']['BGC_sources'] = {}
        result['overall_contamination']['BGC_sources']['mean'] = np_nan_aggregate_w_warning([r['metrics']['BGC_sources'] for r in results], func=np.nanmean, name="BGC_sources")
        result['overall_contamination']['BGC_sources']['std'] = np_nan_aggregate_w_warning([r['metrics']['BGC_sources'] for r in results], func=np.nanstd, name="BGC_sources")
        # Dataset-level contamination matrices: per-cell nanmean of the per-case fg_con_ratios_dict
        # (and its after-thresh variant) across cases where the GT row-class is present, so cells are
        # NOT diluted by the per-case GT-class count (the dilution that shrinks the avg_* metrics).
        # Reuses the DWE confusion machinery (identical {gt: {pred: ratio}} shape). Per matrix we keep
        # the dict-of-dict in the summary JSON and derive two scalars: con_cm_sum (total over all cells)
        # and con_cm_fgfg (fg-fg confusion only, pred != 0; under-seg = col 0 is the remainder).
        from nnunetv2.evaluation.distance_weighted_error import (
            aggregate_confusion, confusion_to_dense, save_confusion_matrix_png)
        con_matrix_dense = {}   # _mat_key -> (dense_array, classes, csv/png filename stub)
        for _src_key, _mat_key, _sp, _stub in [
            ('fg_con_ratios_dict',              'con_matrix',              'con_cm',              'contamination_matrix'),
            ('fg_con_ratios_dict_after_thresh', 'con_matrix_after_thresh', 'con_cm_after_thresh', 'contamination_matrix_after_thresh'),
        ]:
            _M, _classes, _sc = aggregate_confusion([r['metrics'].get(_src_key, {}) for r in results])
            result['overall_contamination'][f'{_sp}_sum']  = {'mean': _sc['dwe_cm_sum'],  'std': 0.0}
            result['overall_contamination'][f'{_sp}_fgfg'] = {'mean': _sc['dwe_cm_fgfg'], 'std': 0.0}
            result['overall_contamination'][_mat_key] = _M   # dict-of-dict; skipped by the scalar txt loop
            con_matrix_dense[_mat_key] = (confusion_to_dense(_M, _classes), _classes, _stub)
        # Save overall contamination metrics as txt file
        if output_file is not None:
            contamination_metrics_file = os.path.join(out_dir, f"overall_contamination_metrics.txt")
            with open(contamination_metrics_file, 'w') as f:
                f.write(f"Overall Contamination Metrics on {len(results)} Samples:\n")
                f.write(f"contamination_config: {result['overall_contamination'].get('contamination_config', {})}\n")
                for key1, v1 in result['overall_contamination'].items():
                    if isinstance(v1, dict) and 'mean' in v1:   # skip the con_matrix dict-of-dicts and contamination_config
                        f.write(f"{key1}: {v1['mean']:.4f} +- {v1['std']:.4f}\n")
            # dataset contamination matrices as their own CSV + log/linear heatmaps. The PNG color
            # scale is pinned to a constant via CONTAMINATION_MATRIX_PNG_VMIN/VMAX (unset -> per-matrix
            # auto) so heatmaps from different models are visually comparable; both the non-thresh and
            # after-thresh matrices share the same scale.
            _con_png_vmin = _env_float('CONTAMINATION_MATRIX_PNG_VMIN')
            _con_png_vmax = _env_float('CONTAMINATION_MATRIX_PNG_VMAX')
            for _mat_key, (_dense, _classes, _stub) in con_matrix_dense.items():
                save_confusion_matrix_as_csv(_dense, _classes,
                                             os.path.join(out_dir, f"{_stub}.csv"),
                                             verbose=verbose, float_format="%.6g")
                for _scale in ('log', 'linear'):
                    try:
                        save_confusion_matrix_png(_dense, _classes,
                                                  os.path.join(out_dir, f"{_stub}_{_scale}.png"),
                                                  scale=_scale, title=f"{_mat_key} ({_scale} scale)",
                                                  vmin=_con_png_vmin, vmax=_con_png_vmax)
                    except Exception as _e:
                        print(f"[compute_metrics_on_filelist] contamination matrix PNG ({_stub},{_scale}) failed: {_e!r}")
            if verbose:
                print(f"[{inspect.currentframe().f_code.co_name}] Saved overall contamination metrics to {contamination_metrics_file}")
                # print the txt file content
                os.system(f"cat {contamination_metrics_file}")

        # Slim `results` before JSON export
        keys_to_remove = [k for k in results[0]['metrics'].keys() if isinstance(k, str) and (k.endswith('_mask') or k.endswith('_interface'))]
        keys_to_remove += ['fg_con_ratios_dict', 'fg_con_ratios_dict_after_thresh']  # we keep the _debug variants in JSON
        if verbose:
            keys_str = '\n\t'.join(keys_to_remove)
            print(f"[{inspect.currentframe().f_code.co_name}] Removing keys from per-case metrics before JSON export to reduce file size:\n\t{keys_str}")
        for r in results:
            r = r['metrics']
            for key in keys_to_remove:
                if key in r:
                    r.pop(key)

    if to_compute_dwe:
        # dataset-level distance-weighted-error metrics. Two scaled sums + two ghost rollups are mean/std
        # over cases. The confusion matrix is nanmean per cell (rows of GT-absent classes excluded), and
        # four scalars are derived from it: dwe_cm_sum and its partition row0 (BG->FG = false positives) /
        # col0 (under-seg) / fgfg (FG<->FG confusion). Per-case dicts (dwe_epp_per_pred_cls,
        # dwe_ghost_cls_nvox, dwe_confusion) stay in summary.json; the params block is lifted once.
        from nnunetv2.evaluation.distance_weighted_error import (
            aggregate_confusion, confusion_to_dense, save_confusion_matrix_png)
        result['overall_dwe'] = {}
        for _dk in ('dwe_under_seg', 'dwe_epp', 'dwe_error', 'dwe_epp_from_bg', 'dwe_epp_fg_conf',
                    'dwe_ghost_nvox', 'dwe_ghost_ncls'):
            result['overall_dwe'][_dk] = {
                'mean': np_nan_aggregate_w_warning([r['metrics'][_dk] for r in results], func=np.nanmean, name=_dk),
                'std':  np_nan_aggregate_w_warning([r['metrics'][_dk] for r in results], func=np.nanstd,  name=_dk),
            }
        cm_M, cm_classes, cm_scalars = aggregate_confusion(
            [r['metrics'].get('dwe_confusion', {}) for r in results])
        for _k, _v in cm_scalars.items():       # store {'mean','std'} so the CSV/txt path stays uniform
            result['overall_dwe'][_k] = {'mean': _v, 'std': 0.0}
        result['overall_dwe']['dwe_confusion'] = cm_M    # dict-of-dict; skipped by the scalar txt loop
        result['dwe_params'] = results[0]['metrics'].get('dwe_params', {})
        for r in results:                       # de-duplicate the params block out of every case
            r['metrics'].pop('dwe_params', None)
        if output_file is not None:
            # overall DWE metrics txt (mirror overall_contamination_metrics.txt); the matrix entry is
            # a dict-of-dict so we only write entries shaped {'mean','std'}.
            dwe_metrics_file = os.path.join(out_dir, "overall_dwe_metrics.txt")
            with open(dwe_metrics_file, 'w') as f:
                f.write(f"Overall Distance-Weighted Error on {len(results)} Samples:\n")
                for key1, v1 in result['overall_dwe'].items():
                    if isinstance(v1, dict) and 'mean' in v1:
                        f.write(f"{key1}: {v1['mean']:.4f} +- {v1['std']:.4f}\n")
                f.write(f"params: {result['dwe_params']}\n")
            # dataset confusion matrix as its own CSV + log/linear heatmaps
            cm_dense = confusion_to_dense(cm_M, cm_classes)
            save_confusion_matrix_as_csv(cm_dense, cm_classes,
                                         os.path.join(out_dir, "dwe_confusion_matrix.csv"),
                                         verbose=verbose, float_format="%.6g")
            # Pin the PNG color scale via DWE_CONFUSION_PNG_VMIN/VMAX (unset -> per-matrix auto) so
            # heatmaps from different models are visually comparable.
            _dwe_png_vmin = _env_float('DWE_CONFUSION_PNG_VMIN')
            _dwe_png_vmax = _env_float('DWE_CONFUSION_PNG_VMAX')
            for _scale in ('log', 'linear'):
                try:
                    save_confusion_matrix_png(cm_dense, cm_classes,
                                              os.path.join(out_dir, f"dwe_confusion_matrix_{_scale}.png"),
                                              scale=_scale, vmin=_dwe_png_vmin, vmax=_dwe_png_vmax)
                except Exception as _e:
                    print(f"[compute_metrics_on_filelist] DWE confusion PNG ({_scale}) failed: {_e!r}")
            if verbose:
                print(f"[{inspect.currentframe().f_code.co_name}] Saved overall DWE metrics to {dwe_metrics_file}")
                os.system(f"cat {dwe_metrics_file}")

    if to_compute_topbrain:
        # TopBrain folder evaluation (clsAvgDice / Betti-0 error / side-road detection F1), scored
        # off the matched file lists with the bundled topbrain25_eval. Driven by topbrain_track
        # ('ct'/'mr'/'auto'); 'auto' splits a mixed CT+MR set by filename and case-weight-combines.
        result['topbrain'] = _run_topbrain(files_ref, files_pred, topbrain_track, num_processes)
        if verbose:
            print(f"[{inspect.currentframe().f_code.co_name}] TopBrain ({topbrain_track}): {result['topbrain']}")

    recursive_fix_for_json_export(result)
    if output_file is not None:
        save_summary_json(result, output_file)
        if verbose:
            print(f"[{inspect.currentframe().f_code.co_name}] Saved summary metrics to {output_file}")

    # Optionally append a single one-line score row to a shared CSV so models can be compared at a glance.
    # Driven entirely by env vars, independent of output_file. The whole column logic lives in
    # append_one_line_score_csv() so it can be reused to regenerate a CSV from saved summary.json files
    # (without re-running evaluation); see houjing_scripts/.../regen_one_line_score_csv.py.
    csv_path = os.environ.get('ONE_LINE_SCORE_CSV_SAVE_PATH', None)
    if csv_path is not None:
        model_alias = os.environ.get('ONE_LINE_SCORE_CSV_MODEL_ALIAS', '')
        append_one_line_score_csv(
            result, csv_path,
            model_alias=model_alias,
            select=os.environ.get('ONE_LINE_SCORE_CSV_COLUMNS', None),
            with_per_class=os.environ.get('ONE_LINE_SCORE_CSV_WITH_PER_CLASS_DICE', None),
            verbose=verbose,
        )
        if verbose:
            print(f"[{inspect.currentframe().f_code.co_name}] Appended score row for '{model_alias}' to {csv_path}")

    if verbose:
        end_time = time.time()
        print(f"[{inspect.currentframe().f_code.co_name}] Finished in {end_time - start_time:.2f} seconds.\n")
    return result

def compute_metrics_roi(reference_file: str, prediction_file: str, 
                    roi_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None, ratio:float = 0.15) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    assert ignore_label is None, "ignore_label not supported for ROI evaluation"

    # Create ROI mask
    with open(roi_file) as f:
        contents = f.readlines()
        size = [int(contents[1].split(" ")[-3]), int(contents[1].split(" ")[-2]), int(contents[1].split(" ")[-1][:-1])]
        locaion = [int(contents[2].split(" ")[-3]), int(contents[2].split(" ")[-2]), int(contents[2].split(" ")[-1][:-1])]

        size = [size[2], size[1], size[0]]
        locaion = [locaion[2], locaion[1], locaion[0]]
        
        # for j in range(3):
        #     locaion[j] = max(0, locaion[j]-int(ratio*size[j]))
        #     size[j] += int(2*ratio*size[j])  # it may run overboard, but it doesn't matter in the following list indexing


    roi_mask = np.zeros_like(seg_pred, dtype=bool)
    roi_mask[:, locaion[0]:locaion[0]+size[0], locaion[1]:locaion[1]+size[1], locaion[2]:locaion[2]+size[2]] = True
    ignore_mask = ~roi_mask # negation of ROI mask since "compute_tp_fp_fn_tn" expects an ignore mask

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)

        # view masks with napari
        #viewer = napari.Viewer()
        #viewer.add_image(mask_ref)
        #viewer.add_image(mask_pred)
        #viewer.add_image(roi_mask)
        #napari.run()

        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
            results['metrics'][r]['clDice'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
            roi_pred_mask = mask_pred & roi_mask
            roi_ref_mask = mask_ref & roi_mask
            results['metrics'][r]['clDice'] = clDice(roi_ref_mask[0], roi_pred_mask[0])
        betti_errors = betti_number_error_all_classes(gt_array=mask_ref[0], pred_array=mask_pred[0])
        results['metrics'][r]['Betti_0_error'] = betti_errors['Betti_0_error']
        results['metrics'][r]['Betti_1_error'] = betti_errors['Betti_1_error']
        results['metrics'][r]['hd95'] = compute_hd95(mask_pred[0].astype(np.int8), mask_ref[0].astype(np.int8))
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    
    # Foreground / Background metrics
    results['fg_bg_metrics'] = {}
    mask_ref = seg_ref > 0
    mask_pred = seg_pred > 0

    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
    if tp + fp + fn == 0:
        results['fg_bg_metrics']['Dice'] = np.nan
        results['fg_bg_metrics']['IoU'] = np.nan
        results['fg_bg_metrics']['clDice'] = np.nan
    else:
        results['fg_bg_metrics']['Dice'] = 2 * tp / (2 * tp + fp + fn)
        results['fg_bg_metrics']['IoU'] = tp / (tp + fp + fn)
        roi_pred_mask = mask_pred & roi_mask
        roi_ref_mask = mask_ref & roi_mask
        results['fg_bg_metrics']['clDice'] = clDice(roi_ref_mask[0], roi_pred_mask[0])
    results['fg_bg_metrics']['FP'] = fp
    results['fg_bg_metrics']['TP'] = tp
    results['fg_bg_metrics']['FN'] = fn
    results['fg_bg_metrics']['TN'] = tn
    results['fg_bg_metrics']['n_pred'] = fp + tp
    results['fg_bg_metrics']['n_ref'] = fn + tp

    return results


def compute_metrics_on_folder_roi(folder_ref: str, folder_pred: str,
                              roi_folder: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    assert os.path.exists(roi_folder), "ROI folder does not exist. Add a folder containing the ROI location & size in .txt \
                                        files to your nnUNet_raw folder. Ask Max about this if in doubt."

    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    files_roi = subfiles(roi_folder, suffix=".txt", join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    files_roi = [join(roi_folder, os.path.basename(file_ref).split(".")[0].replace("whole", "roi") + ".txt") for file_ref in files_ref]
    #for i in list(zip(files_ref, files_pred, files_roi, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
    #    compute_metrics_roi(*i)
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics_roi,
            list(zip(files_ref, files_pred, files_roi, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.nanmean(values)

    # Foreground / Background metrics
    metric_list = list(results[0]['fg_bg_metrics'].keys())
    fg_bg_mean = {}
    for m in metric_list:
        fg_bg_mean[m] = np.nanmean([i['fg_bg_metrics'][m] for i in results])
    

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    recursive_fix_for_json_export(fg_bg_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean, 'fg_bg_mean': fg_bg_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               compute_hd95=False, HD95_UPPER_BOUND=290,
                               nan_as_one=False,
                               to_compute_contamination=False,
                               to_compute_dwe=False, dwe_max_dist=500.0, dwe_norm_by=1e6,
                               dwe_edt_cache_dir=None, dwe_weight_fn='linear', dwe_weight_margin=3.0,
                               gt_hook=None, pred_hook=None,
                               ):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill,
                              compute_hd95=compute_hd95, HD95_UPPER_BOUND=HD95_UPPER_BOUND,
                              nan_as_one=nan_as_one,
                              to_compute_contamination=to_compute_contamination,
                              to_compute_dwe=to_compute_dwe, dwe_max_dist=dwe_max_dist,
                              dwe_norm_by=dwe_norm_by, dwe_edt_cache_dir=dwe_edt_cache_dir,
                              dwe_weight_fn=dwe_weight_fn, dwe_weight_margin=dwe_weight_margin,
                              gt_hook=gt_hook, pred_hook=pred_hook,
                              )


def compute_metrics_on_folder2_roi(folder_ref: str, folder_pred: str, roi_folder: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder_roi(folder_ref, folder_pred, roi_folder, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    """HOUJING: make all arguments start with '--' to be consistent with other entry points"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('--pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('--djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('--pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('--o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('--np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    parser.add_argument('--compute_hd95', action='store_true', help='compute HD95 metric')
    parser.add_argument('--HD95_UPPER_BOUND', type=int, required=False, default=290,
                        help='upper bound for HD95 computation. Default: 290')
    parser.add_argument('--nan_as_one', action='store_true', help='treat NaN Dice as 1')
    parser.add_argument('--compute_contamination', action='store_true', help='compute contamination metrics')
    parser.add_argument('--compute_dwe', action='store_true', help='compute distance-weighted error (DWE) metrics')
    parser.add_argument('--dwe_max_dist', type=float, required=False, default=500.0,
                        help='DWE: punishment distance (voxels) for predicted classes absent from GT. Default: 500')
    parser.add_argument('--dwe_norm_by', type=float, required=False, default=1e6,
                        help='DWE: divisor applied to both per-volume sums. Default: 1e6')
    parser.add_argument('--dwe_edt_cache_dir', type=str, required=False, default=None,
                        help='DWE: dir for the blosc2 EDT cache. Default: <gt_folder>/.dwe_edt_cache')
    parser.add_argument('--dwe_weight_fn', type=str, required=False, default='linear',
                        choices=['linear', 'ones', 'margin_linear', 'margin_ones'],
                        help="DWE weighting (governs sums AND confusion matrix): 'linear' w=d, "
                             "'ones' w=1 (counts), 'margin_linear' w=0 if d<=margin else d, "
                             "'margin_ones' w=0 if d<=margin else 1 (margin-thresholded count). Default: linear")
    parser.add_argument('--dwe_weight_margin', type=float, required=False, default=3.0,
                        help="DWE: margin (voxels) for --dwe_weight_fn margin_linear/margin_ones. Default: 3")
    args = parser.parse_args()
    compute_metrics_on_folder2(
        args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill,
        compute_hd95=args.compute_hd95,
        HD95_UPPER_BOUND=args.HD95_UPPER_BOUND,
        nan_as_one=args.nan_as_one,
        to_compute_contamination=args.compute_contamination,
        to_compute_dwe=args.compute_dwe, dwe_max_dist=args.dwe_max_dist,
        dwe_norm_by=args.dwe_norm_by, dwe_edt_cache_dir=args.dwe_edt_cache_dir,
        dwe_weight_fn=args.dwe_weight_fn, dwe_weight_margin=args.dwe_weight_margin,
    )


def evaluate_folder_roi_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('roi_folder', type=str, help='folder with ROI coordinates')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2_roi(args.gt_folder, args.pred_folder, args.roi_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)
    

def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)
