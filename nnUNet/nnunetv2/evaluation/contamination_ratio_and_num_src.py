from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from pprint import pprint
import shutil
import os

def to_numpy(img):
    """Convert SimpleITK Image or path-like to NumPy array if needed."""
    input_type = type(img)
    if isinstance(img, (str, Path)):
        img = sitk.ReadImage(str(img))
    if isinstance(img, sitk.Image):
        return sitk.GetArrayFromImage(img)
    assert isinstance(img, np.ndarray), f"Unsupported type {input_type}"
    return img

def clean_numpy(obj):
    """Transform numpy types in obj to native Python types for better print readability."""
    if isinstance(obj, dict):
        return {clean_numpy(k): clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(clean_numpy(x) for x in obj)
    else:
        return obj

def roll_zeros(a, shift, axis=None):
    """The same as np.roll, but fills the emptied positions with zeros instead of wrapping around."""
    out = np.zeros_like(a)

    if axis is None:
        a = a.ravel()
        out = out.ravel()
        axis = 0

    n = a.shape[axis]
    if shift == 0:
        return a.copy()

    shift = int(shift)
    if shift > 0:
        sl_src = [slice(None)] * a.ndim
        sl_dst = [slice(None)] * a.ndim
        sl_src[axis] = slice(0, n - shift)
        sl_dst[axis] = slice(shift, n)
    else:
        shift = -shift
        sl_src = [slice(None)] * a.ndim
        sl_dst = [slice(None)] * a.ndim
        sl_src[axis] = slice(shift, n)
        sl_dst[axis] = slice(0, n - shift)

    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out

def all_interfaces_26_no_bg(L, bg=0, margin=1):
    """Return a boolean array marking all inter-label interface voxels in L using 26-connectivity.
    Not consider interface between any label and the background.
    margin: bigger value means thicker interface region.
    """
    L = np.asarray(L)
    out = np.zeros_like(L, dtype=bool)

    fg = (L != bg)

    for dx in (-margin, 0, margin):
        for dy in (-margin, 0, margin):
            for dz in (-margin, 0, margin):
                if dx==0 and dy==0 and dz==0:
                    continue

                S = roll_zeros(roll_zeros(roll_zeros(L, dx, 0), dy, 1), dz, 2)
                out |= fg & (S != bg) & (S != L)

    return out

def compute_contamination(gt, pred, fg_fg_interface_margin=0, bg_surface_margin=0,
                          UnderSeg_ratio_thresh=0, FGC_ratio_thresh=0, return_masks=False):
    """
    1) Compute area ratio of foreground labels being contaminated, 
       by background or any other foreground label,
       averaged over existing foreground labels.
    2) Compute number of contamination sources for foreground labels,
       averaged over existing foreground labels.
    3) Compute total number of background voxels contaminated as foreground labels.
    4) Compute number of contamination sources for background.

    Parameters
    ----------
    gt : np.ndarray or sitk.Image or path-like
        Ground-truth label map.
    pred : np.ndarray or sitk.Image or path-like
        Predicted label map.
    fg_fg_interface_margin : int, default 0
        Margin for fg-fg interface detection.
        margin=0: no relaxation
        margin=1: each fg voxel adjacent to another label is considered interface.
        margin=k: each fg voxel within k-voxel distance to another label is considered interface.
    bg_surface_margin : int, default 0
        Margin for bg surface detection.
        margin=0: no relaxation
        margin=1: each bg voxel adjacent to any fg voxel is considered surface.
        margin=k: each bg voxel within k-voxel distance to any fg voxel is considered surface.
    UnderSeg_ratio_thresh : float, default 0
        Ratio threshold for the bg column (pred_cls == 0, i.e. under-segmentation) of the
        `*_after_thresh` metrics and dicts. Only bg-column entries with `ratio > UnderSeg_ratio_thresh`
        are counted toward the `UnderSeg_*_after_thresh` metrics / recorded in the `*_after_thresh`
        dicts. Default 0 → every real (nonzero) bg-column entry passes.
    FGC_ratio_thresh : float, default 0
        Ratio threshold for the other-fg columns (pred_cls != 0, i.e. confusion with other fg classes)
        of the `*_after_thresh` metrics and dicts. Only fg-column entries with
        `ratio > FGC_ratio_thresh` are counted toward the `FGC_*_after_thresh` metrics /
        recorded in the `*_after_thresh` dicts. Default 0 → every real (nonzero) fg-column entry passes.
    return_masks : bool, default False
        Whether to return the masks used for calculation.

    Returns
    -------    
    dict {
        "contamination_config": dict
            - metric configuration:
              fg_fg_interface_margin, bg_surface_margin, UnderSeg_ratio_thresh, FGC_ratio_thresh

        "fg_con_ratios_dict": a dict {gt_class: {error_pred_class: area_ratio, ...}, ...}
            - If no gt class, the whole dict is empty.
            - For each gt class, the inner dict maps each mistakenly predicted class to its contamination area ratio.
            - If a gt class is not contaminated, the inner dict is empty.
        
        "fg_con_ratios_dict_debug": a dict {gt_class: {error_pred_class: {"n_error_voxels": int, "n_gt_voxels": int, "ratio": float}, ...}, ...}
            - Same structure as "fg_con_ratios_dict", but save a dict for debugging.

        "fg_con_ratios_dict_after_thresh": same structure as "fg_con_ratios_dict",
            - but only keeps entries whose ratio passes its column's threshold: bg column (pred 0) uses
              UnderSeg_ratio_thresh, other-fg columns use FGC_ratio_thresh.

        "fg_con_ratios_dict_debug_after_thresh": same structure as "fg_con_ratios_dict_debug",
            - but only keeps entries whose ratio passes its column's threshold (see above).

        "fg_avg_con_ratio": float
            - average contamination ratio across all foreground classes in GT
        
        "fg_avg_con_sources": float
            - average number of contamination sources across all foreground classes in GT

        "UnderSeg_ratio": float
            - the bg column of fg_con_ratios_dict (pred class 0), i.e. under-segmentation:
              average ratio of fg classes lost to background, across all fg classes in GT

        "UnderSeg_prevalence": float
            - average number of under-segmentation sources (i.e. avg appearance of the bg source)
              across all fg classes in GT; each fg class contributes 0 or 1

        "FGC_ratio": float
            - the non-bg columns of fg_con_ratios_dict, i.e. contamination by other fg classes:
              average ratio across all fg classes in GT

        "FGC_sources": float
            - average number of contamination sources that are other fg classes, across all fg classes in GT

        Note: UnderSeg_ratio + FGC_ratio == fg_avg_con_ratio, and likewise for sources/prevalence.

        "UnderSeg_ratio_after_thresh": float
        "UnderSeg_prevalence_after_thresh": float
        "FGC_ratio_after_thresh": float
        "FGC_sources_after_thresh": float
            - Same as the four UnderSeg/FGC metrics above, but only counting matrix entries whose
              ratio passes its column's threshold (bg column: UnderSeg_ratio_thresh; other-fg columns:
              FGC_ratio_thresh), i.e. substantial contamination only.

        "BGC_voxels_dict": a dict {error_pred_class: number_of_voxels, ...}
            - Number of bg voxels predicted as each fg class
            - If the gt bg is not contaminated, the dict is empty.
        
        "BGC_voxels": float
            - sum of contamination voxels
        
        "BGC_sources": float
            - number of contamination sources
        
        Only returned if return_masks=True:
        -----------------------------------

        "fg_fg_interface_mask": np.ndarray (boolean array of same shape as input),
            - boolean mask marking interface voxels between fg labels, which are ignored in error counting
        
        "bg_surface_mask": np.ndarray (boolean array of same shape as input),
            - boolean mask marking surface voxels of bg, which are ignored in error counting
        
        "non_interface_mask": np.ndarray (boolean array of same shape as input),
            - boolean mask marking voxels that are not on fg-fg interface nor bg surface
        
        "fg_error_mask": np.ndarray (boolean array of same shape as input)
            - boolean mask marking error voxels on gt fg region (GT != Pred) excluding fg-fg interface voxels
        
        "bg_error_mask": np.ndarray (boolean array of same shape as input)
            - boolean mask marking error voxels on gt bg region (GT != Pred) excluding bg surface voxels
        
        "gt_erased_interface": np.ndarray (same shape as input)
            - copy of GT with fg-fg interface voxels erased (set to 0)

        "pred_erased_interface": np.ndarray (same shape as input)
            - copy of Pred with fg-fg interface voxels erased (set to 0)
    }
    """
    gt = to_numpy(gt)
    pred = to_numpy(pred)

    assert gt.shape == pred.shape, f"GT and prediction must have the same shape, but got {gt.shape} vs {pred.shape}"
    assert len(gt.shape) == 3, f"Only 3D images are supported, but got shape {gt.shape}. You can add an extra dimension to make it 3D."

    # Use 26-connectivity (3x3x3 neighborhood)
    fg_fg_interface_mask = all_interfaces_26_no_bg(gt, margin=fg_fg_interface_margin)
    # Be Careful: if iterations=0, binary_dilation repeats until the result does not change anymore
    # So we only call binary_dilation if margin > 0
    if bg_surface_margin > 0:        
        bg_surface_mask = binary_dilation(gt != 0, structure=generate_binary_structure(3, 3), iterations=bg_surface_margin) & (gt == 0)
    else:
        bg_surface_mask = np.zeros_like(gt, dtype=bool)
    non_interface_mask = ~ (fg_fg_interface_mask | bg_surface_mask)

    if return_masks:
        gt_erased_interface = gt.copy()
        gt_erased_interface[fg_fg_interface_mask] = 0
        pred_erased_interface = pred.copy()
        pred_erased_interface[fg_fg_interface_mask] = 0
    if not return_masks:
        del fg_fg_interface_mask
        del bg_surface_mask

    # Error masks after relaxation
    error_mask = (gt != pred) & non_interface_mask
    fg_error_mask = (gt != 0) & error_mask
    bg_error_mask = (gt == 0) & error_mask
    del error_mask

    # n_gt_voxels per class
    classes_b4_relax, n_voxels_b4_relax = np.unique(gt, return_counts=True)
    classes, n_voxels = np.unique(gt[non_interface_mask], return_counts=True)
    tmp = {cls: n for cls, n in zip(classes, n_voxels)}
    
    # IMPORTANT: `cls` is before relaxation, while `tmp.get(cls, 0)` is after relaxation.
    # If value is 0 for a key, it means all voxels of that class are on the interface.
    # We still want to keep this class in the dict, solely to show which classes are in unrelaxed GT.
    # The denominator for fg_con_ratios_dict is also number of GT fg classes before relaxation.
    cls_to_n_gt_voxels = {cls: tmp.get(cls, 0) for cls in classes_b4_relax}

    # sort classes for consistent ordering
    cls_to_n_gt_voxels = dict(sorted(cls_to_n_gt_voxels.items()))

    if not return_masks:
        del non_interface_mask

    ########################################
    # Foreground contamination
    ########################################
    
    # For each fg class in GT, make an entry in the result dict
    fg_con_ratios_dict = {k: {} for k in cls_to_n_gt_voxels if k != 0}
    fg_con_ratios_dict_debug = {k: {} for k in cls_to_n_gt_voxels if k != 0}
    # Same matrices, but only keeping entries whose ratio passes its column's threshold
    # (bg column: UnderSeg_ratio_thresh; other-fg columns: FGC_ratio_thresh).
    fg_con_ratios_dict_after_thresh = {k: {} for k in cls_to_n_gt_voxels if k != 0}
    fg_con_ratios_dict_debug_after_thresh = {k: {} for k in cls_to_n_gt_voxels if k != 0}
    n_gt_classes_fg = len(fg_con_ratios_dict)
    gt_fg_exists = n_gt_classes_fg > 0
    gt_bg_exists = 0 in cls_to_n_gt_voxels

    sum_ratio = 0
    sum_sources = 0
    # Split the totals by contamination source: bg (pred 0, i.e. under-segmentation) vs other fg classes.
    sum_UnderSeg_ratio = 0
    sum_UnderSeg_prevalence = 0
    sum_FGC_ratio = 0
    sum_FGC_sources = 0
    # Same split totals, but only counting entries whose ratio passes its column's threshold
    # (bg column: UnderSeg_ratio_thresh; other-fg columns: FGC_ratio_thresh).
    sum_UnderSeg_ratio_after_thresh = 0
    sum_UnderSeg_prevalence_after_thresh = 0
    sum_FGC_ratio_after_thresh = 0
    sum_FGC_sources_after_thresh = 0

    # We first find the mistakenly predicted classes
    # Then for each mistakenly predicted class, we calculate how many voxels come from each GT class
    # Equivalent to a loop over GT classes and `np.unique` count error predictions for each GT class
    
    # error_pred_classes could be empty
    error_pred_classes = np.unique(pred[fg_error_mask])  # include bg class
    # sort classes for consistent ordering
    error_pred_classes = np.sort(error_pred_classes)
    for pred_cls in error_pred_classes:
        gt_classes, n_error_voxels = np.unique(gt[(pred == pred_cls) & fg_error_mask], return_counts=True)
        for gt_cls, count in zip(gt_classes, n_error_voxels):
            assert pred_cls != gt_cls, f"pred_cls {pred_cls} should not equal gt_cls {gt_cls}"
            assert gt_cls in fg_con_ratios_dict, f"{gt_cls} not in {fg_con_ratios_dict}"
            ratio = count / cls_to_n_gt_voxels[gt_cls]
            fg_con_ratios_dict[gt_cls][pred_cls] = ratio
            fg_con_ratios_dict_debug[gt_cls][pred_cls] = {
                "n_error_voxels": int(count),
                "n_gt_voxels": int(cls_to_n_gt_voxels[gt_cls]),
                "ratio": float(ratio),
            }
            sum_ratio += ratio
            sum_sources += 1
            if pred_cls == 0:
                sum_UnderSeg_ratio += ratio
                sum_UnderSeg_prevalence += 1
            else:
                sum_FGC_ratio += ratio
                sum_FGC_sources += 1
            # Thresholded bookkeeping: only entries with substantial contamination (ratio > thresh).
            # The bg column (under-segmentation) and the other-fg columns get separate thresholds.
            thresh = UnderSeg_ratio_thresh if pred_cls == 0 else FGC_ratio_thresh
            if ratio > thresh:
                fg_con_ratios_dict_after_thresh[gt_cls][pred_cls] = ratio
                fg_con_ratios_dict_debug_after_thresh[gt_cls][pred_cls] = {
                    "n_error_voxels": int(count),
                    "n_gt_voxels": int(cls_to_n_gt_voxels[gt_cls]),
                    "ratio": float(ratio),
                }
                if pred_cls == 0:
                    sum_UnderSeg_ratio_after_thresh += ratio
                    sum_UnderSeg_prevalence_after_thresh += 1
                else:
                    sum_FGC_ratio_after_thresh += ratio
                    sum_FGC_sources_after_thresh += 1

    fg_avg_con_ratio = sum_ratio / n_gt_classes_fg if gt_fg_exists else np.nan
    fg_avg_con_sources = sum_sources / n_gt_classes_fg if gt_fg_exists else np.nan
    UnderSeg_ratio = sum_UnderSeg_ratio / n_gt_classes_fg if gt_fg_exists else np.nan
    UnderSeg_prevalence = sum_UnderSeg_prevalence / n_gt_classes_fg if gt_fg_exists else np.nan
    FGC_ratio = sum_FGC_ratio / n_gt_classes_fg if gt_fg_exists else np.nan
    FGC_sources = sum_FGC_sources / n_gt_classes_fg if gt_fg_exists else np.nan
    UnderSeg_ratio_after_thresh = sum_UnderSeg_ratio_after_thresh / n_gt_classes_fg if gt_fg_exists else np.nan
    UnderSeg_prevalence_after_thresh = sum_UnderSeg_prevalence_after_thresh / n_gt_classes_fg if gt_fg_exists else np.nan
    FGC_ratio_after_thresh = sum_FGC_ratio_after_thresh / n_gt_classes_fg if gt_fg_exists else np.nan
    FGC_sources_after_thresh = sum_FGC_sources_after_thresh / n_gt_classes_fg if gt_fg_exists else np.nan

    if not return_masks:
        del fg_error_mask
            
    ########################################
    # Background contamination
    ########################################

    # error_pred_classes could be empty
    error_pred_classes, n_error_voxels = np.unique(pred[bg_error_mask], return_counts=True)
    assert 0 not in error_pred_classes, "Background class should not be counted as contamination source for background."
    BGC_voxels_dict = {k: v for k, v in zip(error_pred_classes, n_error_voxels)}
    # sort classes for consistent ordering
    BGC_voxels_dict = dict(sorted(BGC_voxels_dict.items()))
    BGC_voxels = sum(n_error_voxels) if gt_bg_exists else np.nan
    BGC_sources = len(BGC_voxels_dict) if gt_bg_exists else np.nan

    res = {
        "contamination_config": {
            "fg_fg_interface_margin": int(fg_fg_interface_margin),
            "bg_surface_margin": int(bg_surface_margin),
            "UnderSeg_ratio_thresh": float(UnderSeg_ratio_thresh),
            "FGC_ratio_thresh": float(FGC_ratio_thresh),
        },
        "fg_con_ratios_dict": fg_con_ratios_dict,
        "fg_con_ratios_dict_debug": fg_con_ratios_dict_debug,
        "fg_con_ratios_dict_after_thresh": fg_con_ratios_dict_after_thresh,
        "fg_con_ratios_dict_debug_after_thresh": fg_con_ratios_dict_debug_after_thresh,
        "fg_avg_con_ratio": fg_avg_con_ratio,
        "fg_avg_con_sources": fg_avg_con_sources,
        "UnderSeg_ratio": UnderSeg_ratio,
        "UnderSeg_prevalence": UnderSeg_prevalence,
        "FGC_ratio": FGC_ratio,
        "FGC_sources": FGC_sources,
        "UnderSeg_ratio_after_thresh": UnderSeg_ratio_after_thresh,
        "UnderSeg_prevalence_after_thresh": UnderSeg_prevalence_after_thresh,
        "FGC_ratio_after_thresh": FGC_ratio_after_thresh,
        "FGC_sources_after_thresh": FGC_sources_after_thresh,
        "BGC_voxels_dict": BGC_voxels_dict,
        "BGC_voxels": BGC_voxels,
        "BGC_sources": BGC_sources,
    }
    if return_masks:
        res.update({
            "fg_fg_interface_mask": fg_fg_interface_mask,
            "bg_surface_mask": bg_surface_mask,
            "fg_error_mask": fg_error_mask,
            "bg_error_mask": bg_error_mask,
            "gt_erased_interface": gt_erased_interface,
            "pred_erased_interface": pred_erased_interface,
        })
    return res

def test_one_case(gt, pred, out_dir=None, expected=None, verbose=True, title='test', **kwargs):
    import json

    print(f"\n=== Running {title} ===\n")
    res = compute_contamination(gt, pred, **kwargs)
    res = clean_numpy(res)
    mask_keys = [key for key in res if key.endswith('_mask')] + ['gt_erased_interface', 'pred_erased_interface']
    if expected is not None:
        expected = clean_numpy(expected)
        assert res == expected, f"Expected {expected}, but got {res}"
    
    if verbose:
        pprint({k: v for k, v in res.items() if k not in mask_keys})

    # If out_dir is specified, turn masks into uint8, save to NIFTI, in the same space as GT
    if out_dir is not None:
        assert isinstance(gt, (str, Path, sitk.Image)), "GT must be path-like or sitk.Image to save masks in the same space."
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        gt_sitk = sitk.ReadImage(str(gt)) if not isinstance(gt, sitk.Image) else gt        
        for key in mask_keys:
            if key not in res:
                continue
            mask = res[key].astype(np.uint8)  # boolean to uint8
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(gt_sitk)
            sitk.WriteImage(mask_sitk, str(out_dir / f"{key}.nii.gz"))
        
        # Save a json summary as well
        with open(out_dir / "summary.json", 'w') as f:
            json.dump({k: v for k, v in res.items() if k not in mask_keys}, f, indent=4)
        
        # Copy GT and Pred as well
        # Since they are either path-like or sitk.Image, we can always read and write them
        if isinstance(pred, sitk.Image):
            sitk.WriteImage(pred, str(out_dir / "pred.nii.gz"))
        else:
            shutil.copy(str(pred), str(out_dir / "pred.nii.gz"))
        if isinstance(gt, sitk.Image):
            sitk.WriteImage(gt, str(out_dir / "gt.nii.gz"))
        else:
            shutil.copy(str(gt), str(out_dir / "gt.nii.gz"))
    return res

if __name__ == "__main__":
    """
    python contamination_ratio_and_num_src.py
    """
    return_masks = True
    test_dir = './contamination_ratio_and_n_src_test_cases'
    kwargs = {
        'fg_fg_interface_margin': int(os.environ.get('CONTAMINATION_FG_FG_INTERFACE_MARGIN', 3)),
        'bg_surface_margin': int(os.environ.get('CONTAMINATION_BG_SURFACE_MARGIN', 5)),
        'UnderSeg_ratio_thresh': float(os.environ.get('CONTAMINATION_UNDERSEG_RATIO_THRESH', 0)),
        'FGC_ratio_thresh': float(os.environ.get('CONTAMINATION_FGC_RATIO_THRESH', 0)),
    }
    _tmp_dir = os.environ.get('CONTAMINATION_TMP_DIR', './tmp/contamination_ratio_and_n_src_test_outputs')
    out_dir_base = f'{_tmp_dir}/fgfg_margin_{kwargs["fg_fg_interface_margin"]}__bg_margin_{kwargs["bg_surface_margin"]}__UnderSeg_thresh_{kwargs["UnderSeg_ratio_thresh"]}__FGC_thresh_{kwargs["FGC_ratio_thresh"]}'
    test_one_case(f'{test_dir}/lps_8x4x1_GT.nii.gz', f'{test_dir}/lps_8x4x1_Pred_A.nii.gz', out_dir=f'{out_dir_base}/8x4x1_Pred_A', title='8x4x1 Pred_A', verbose=True, return_masks=return_masks, **kwargs)
    test_one_case(f'{test_dir}/lps_8x4x1_GT.nii.gz', f'{test_dir}/lps_8x4x1_Pred_B.nii.gz', out_dir=f'{out_dir_base}/8x4x1_Pred_B', title='8x4x1 Pred_B', verbose=True, return_masks=return_masks, **kwargs)
    test_one_case(f'{test_dir}/lps_8x4x1_GT.nii.gz', f'{test_dir}/lps_8x4x1_Pred_C.nii.gz', out_dir=f'{out_dir_base}/8x4x1_Pred_C', title='8x4x1 Pred_C', verbose=True, return_masks=return_masks, **kwargs)
    test_one_case(f'{test_dir}/lps_11x7x1_GT.nii.gz', f'{test_dir}/lps_11x7x1_Pred.nii.gz', out_dir=f'{out_dir_base}/11x7x1_Pred', title='11x7x1 Pred', verbose=True, return_masks=return_masks, **kwargs)
