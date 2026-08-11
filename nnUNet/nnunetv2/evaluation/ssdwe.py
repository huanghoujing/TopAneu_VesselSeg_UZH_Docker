"""Super-Simplified Distance-Weighted Error (ssDWE).

The smallest member of the DWE family. Where DWE splits error voxels into under-segmentation and
error-predicted-positive (and a confusion matrix), and sDWE keeps the two/three scalars, ssDWE treats
**background and foreground the same** and reports just:

  1. `ssdwe_error` -- one distance-weighted sum over ALL error voxels (gt != pred), and
  2. ghost diagnostics -- predicted classes absent from the GT.

One unified rule for every error voxel (distance in **voxel units**): weight it by the distance to the
nearest TRUE region of its **predicted** label `c`, via `edt(gt != c)`.
- `pred == 0` -> `edt(gt != 0)` = distance to nearest true background (what DWE calls under-seg).
- `pred == c != 0` -> `edt(gt != c)` = distance to nearest true class-c (what DWE calls EPP).
- if `c` is ABSENT from GT (a "ghost" class) there is no distance, so each such voxel is punished with
  `max_dist`.

Because the only difference from DWE is that the under-seg and EPP sums are not kept apart,
`ssdwe_error == dwe_under_seg + dwe_epp` exactly (cross-checked in the self-test).

Shared internals (cuCIM/scipy EDT backend, blosc2 uint16 cache, fixed-point quantization, weight
functions) are imported from `distance_weighted_error` -- single source of truth; reuse
`precompute_dwe_cache_for_files` for the single-process GPU pre-pass.
"""
import numpy as np

from nnunetv2.evaluation.contamination_ratio_and_num_src import to_numpy, clean_numpy
from nnunetv2.evaluation.distance_weighted_error import (
    DWE_SCALE,
    _edt_backend,
    load_or_compute_edt,
    precompute_dwe_cache_for_files,   # re-exported for the GPU pre-pass
    resolve_weight_fn,
    weight_fn_is_distance_free,
)


# ---- the super-simplified metric ------------------------------------------------------------------
def compute_super_simplified_dwe(gt, pred, weight_fn=None, weight_margin=3.0, max_dist=500.0,
                                 dwe_norm_by=1e6, edt_cache_dir=None, gt_key=None, gt_mtime=None,
                                 allow_gpu=True, return_masks=False):
    """One distance-weighted sum over every error voxel (bg/fg treated alike) + ghost diagnostics.
    Distances/weighting/caching match `compute_distance_weighted_error`; only the under-seg vs EPP
    split (and the confusion matrix) are dropped. `ssdwe_error == dwe_under_seg + dwe_epp`.

    weight_fn:     None/'linear'/'ones'/'margin_linear'/'margin_ones' or a callable (see resolve_weight_fn).
    weight_margin: margin (voxels) for the margin variants.
    max_dist:      punishment distance (voxels) for error voxels of a GT-absent ("ghost") class.
    dwe_norm_by:   divisor applied to the reported sum.
    edt_cache_dir / gt_key / gt_mtime: enable the blosc2 EDT cache (gt_key is the GT file path).
    allow_gpu:     GPU EDT is fork-unsafe; pass False inside multiprocess workers (cache reads/scipy).
    """
    gt = to_numpy(gt)
    pred = to_numpy(pred)
    wf, wf_name = resolve_weight_fn(weight_fn, weight_margin)

    error = gt != pred

    # weight_fn='ones' ignores the distance, so skip every EDT compute/load (feed a dummy zero array of
    # the right length). The caller likewise skips the GPU pre-pass. See weight_fn_is_distance_free.
    distance_free = weight_fn_is_distance_free(weight_fn)

    def _edt_c(c):
        return load_or_compute_edt(gt, c, edt_cache_dir, gt_key, gt_mtime, allow_gpu)

    # one unified loop over every predicted label among the error voxels (including pred == 0)
    sum_err = 0.0
    per_cls = {}
    ghost = {}
    for c in np.unique(pred[error]):
        c = int(c)
        sel = error & (pred == c)
        n = int(sel.sum())
        absent = not bool(np.any(gt == c))     # class absent from GT -> "ghost" (c == 0 is never absent)
        if absent:
            ghost[c] = n
        if distance_free:
            w = wf(np.zeros(n))
        elif absent:
            w = wf(np.full(n, float(max_dist)))
        else:
            w = wf(_edt_c(c)[sel])
        s_c = float(w.sum()) / dwe_norm_by
        sum_err += s_c
        per_cls[c] = s_c

    res = {
        "ssdwe_error": sum_err,                          # distance-weighted sum over ALL error voxels
        "ssdwe_error_per_pred_cls": per_cls,             # scaled, debug (summary only); key 0 = under-seg
        "ssdwe_ghost_cls_nvox": ghost,                   # {c: nvox} per ghost class (summary only)
        "ssdwe_ghost_nvox": int(sum(ghost.values())),    # total ghost voxels this case
        "ssdwe_ghost_ncls": int(len(ghost)),             # number of ghost classes this case
        "ssdwe_params": {
            "max_dist": float(max_dist),
            "dwe_norm_by": float(dwe_norm_by),
            "weight_fn": wf_name,
            "weight_margin": float(weight_margin),
            "edt_scale": DWE_SCALE,
            "edt_backend": _edt_backend(allow_gpu),
        },
    }
    if return_masks:
        res["error_mask"] = error
    return res


# ---- self-test ------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Self-test:
        cd /mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW
        python nnunetv2/evaluation/ssdwe.py
    """
    from pprint import pprint
    from nnunetv2.evaluation.distance_weighted_error import compute_distance_weighted_error

    NORM = 1.0
    MAXD = 10.0

    # ---- Test 1: bg + fg errors summed in one number ----
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    gt[0, 3:8, 12:17] = 2
    pred = gt.copy()
    pred[0, 5, 5] = 0          # under-seg of class 1 (deep)  -> dist-to-BG = 3
    pred[0, 5, 11] = 2         # EPP-2 on background (gt 0)   -> dist-to-true-2 = 1
    r = compute_super_simplified_dwe(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 1: unified error sum ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert abs(r["ssdwe_error"] - (3.0 + 1.0)) < 1e-5, r["ssdwe_error"]
    assert r["ssdwe_ghost_ncls"] == 0

    # ---- Test 2: ghost class (absent from GT) punished at max_dist ----
    gt = np.zeros((1, 11, 11), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    pred = gt.copy()
    pred[0, 0, 0] = 5
    pred[0, 0, 1] = 5
    pred[0, 0, 2] = 5
    pred[0, 1, 0] = 5          # 4 ghost voxels of class 5
    r = compute_super_simplified_dwe(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 2: ghost class ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert r["ssdwe_ghost_cls_nvox"] == {5: 4} and r["ssdwe_ghost_nvox"] == 4 and r["ssdwe_ghost_ncls"] == 1
    assert abs(r["ssdwe_error"] - 4 * MAXD) < 1e-5

    # ---- Test 3: cross-check ssdwe_error == dwe_under_seg + dwe_epp, all weight fns ----
    print("\n=== Test 3: cross-check vs distance_weighted_error ===")
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    gt[0, 3:8, 12:17] = 2
    pred = gt.copy()
    pred[0, 5, 5] = 0          # under-seg
    pred[0, 5, 11] = 2         # epp from bg
    pred[0, 4, 4] = 2          # epp fg-conf
    pred[0, 0, 0] = 7          # ghost class 7
    for wfn in ('linear', 'ones', 'margin_linear', 'margin_ones'):
        s = compute_super_simplified_dwe(gt, pred, weight_fn=wfn, weight_margin=3.0, max_dist=MAXD, dwe_norm_by=NORM)
        d = compute_distance_weighted_error(gt, pred, weight_fn=wfn, weight_margin=3.0, max_dist=MAXD, dwe_norm_by=NORM)
        assert abs(s["ssdwe_error"] - (d["dwe_under_seg"] + d["dwe_epp"])) < 1e-9, (wfn, s, d)
        assert s["ssdwe_ghost_cls_nvox"] == d["dwe_ghost_cls_nvox"], (wfn, s, d)
        print(f"  {wfn:13s}: ssdwe_error={s['ssdwe_error']:.3f}  == dwe_under_seg+dwe_epp="
              f"{d['dwe_under_seg'] + d['dwe_epp']:.3f}  OK")

    print("\nAll ssDWE self-tests passed.")
