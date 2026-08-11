"""Simplified Distance-Weighted Error (sDWE).

A matrix-free cousin of `distance_weighted_error.py` (DWE). It computes the same distance-weighted
error mass per volume, but reports it as a handful of **scalars** instead of a full per-(gt, pred)
confusion matrix -- so there is no N x N matrix, no heatmap PNG, no matrix CSV, and no redraw step.

Per error voxel (gt != pred), distance in **voxel units** (same definition as DWE):
- Under-segmentation (pred == 0, gt == g FG): weight by distance to nearest TRUE BACKGROUND,
  D_bg = edt(gt != 0), indexed at the under-seg voxels.
- Error predicted positive / EPP (pred == c != 0, gt != c): weight by distance to nearest TRUE
  class-c, D_c = edt(gt != c). A "ghost" class (c absent from GT) is punished with `max_dist`.

The EPP mass is additionally split, cheaply (no matrix), by the true label under each EPP voxel:
- `epp_from_bg`  -- gt == 0 (classic false positive from background),
- `epp_fg_conf`  -- gt != 0 (FG<->FG confusion).
So the volume's total error decomposes 3 ways: `under_seg + epp_from_bg + epp_fg_conf == under_seg + epp`.
(These three equal the DWE confusion matrix's col0 / row0 / fgfg scalars -- this module just skips
building the matrix to get them.)

Shared internals -- the cuCIM/scipy EDT backend, the blosc2 uint16 disk cache, the fixed-point
quantization, and the weight functions -- are imported from `distance_weighted_error` (single source of
truth; reuse `precompute_dwe_cache_for_files` for the single-process GPU pre-pass too).
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


# ---- the simplified metric ------------------------------------------------------------------------
def compute_simplified_dwe(gt, pred, weight_fn=None, weight_margin=3.0, max_dist=500.0,
                           dwe_norm_by=1e6, edt_cache_dir=None, gt_key=None, gt_mtime=None,
                           allow_gpu=True, return_masks=False):
    """Matrix-free Distance-Weighted Error for one (gt, pred) pair. Returns a dict of scalars (see
    module docstring). Distances/weighting/caching match `compute_distance_weighted_error`; only the
    per-(gt,pred) confusion matrix is dropped, replaced by the EPP from-bg / fg-conf split.

    weight_fn:     None/'linear'/'ones'/'margin_linear'/'margin_ones' or a callable (see
                   resolve_weight_fn). weight_margin: margin (voxels) for the margin variants.
    max_dist:      punishment distance (voxels) for EPP voxels of a GT-absent ("ghost") class.
    dwe_norm_by:   divisor applied to every reported sum.
    edt_cache_dir / gt_key / gt_mtime: enable the blosc2 EDT cache (gt_key is the GT file path).
    allow_gpu:     GPU EDT is fork-unsafe; pass False inside multiprocess workers (cache reads/scipy).
    """
    gt = to_numpy(gt)
    pred = to_numpy(pred)
    wf, wf_name = resolve_weight_fn(weight_fn, weight_margin)

    error = gt != pred
    under = error & (pred == 0)               # under-segmentation
    epp = error & (pred != 0)                 # error predicted positive

    # weight_fn='ones' ignores the distance, so skip every EDT compute/load (feed a dummy zero array of
    # the right length). The caller likewise skips the GPU pre-pass. See weight_fn_is_distance_free.
    distance_free = weight_fn_is_distance_free(weight_fn)

    def _edt_c(c):
        return load_or_compute_edt(gt, c, edt_cache_dir, gt_key, gt_mtime, allow_gpu)

    # under-seg
    if under.any():
        n = int(under.sum())
        d = np.zeros(n) if distance_free else _edt_c(0)[under]
        sum_under = float(wf(d).sum()) / dwe_norm_by
    else:
        sum_under = 0.0

    # epp, split by the true label (gt==0 -> from background, gt!=0 -> FG<->FG confusion)
    sum_epp_from_bg = 0.0
    sum_epp_fg_conf = 0.0
    per_cls = {}
    ghost = {}
    for c in np.unique(pred[epp]):
        c = int(c)
        sel = epp & (pred == c)
        n = int(sel.sum())
        absent = not bool(np.any(gt == c))
        if absent:                             # class absent from GT -> "ghost" class
            ghost[c] = n
        if distance_free:
            w = wf(np.zeros(n))
        elif absent:
            w = wf(np.full(n, float(max_dist)))
        else:
            w = wf(_edt_c(c)[sel])
        from_bg = gt[sel] == 0                 # split this class's EPP weights by true label
        s_bg = float(w[from_bg].sum()) / dwe_norm_by
        s_fg = float(w[~from_bg].sum()) / dwe_norm_by
        sum_epp_from_bg += s_bg
        sum_epp_fg_conf += s_fg
        per_cls[c] = s_bg + s_fg

    sum_epp = sum_epp_from_bg + sum_epp_fg_conf

    res = {
        "sdwe_under_seg": sum_under,
        "sdwe_epp": sum_epp,
        "sdwe_epp_from_bg": sum_epp_from_bg,             # EPP where gt == 0 (false positive from BG)
        "sdwe_epp_fg_conf": sum_epp_fg_conf,             # EPP where gt != 0 (FG<->FG confusion)
        "sdwe_epp_per_pred_cls": per_cls,                # scaled, debug (summary only)
        "sdwe_ghost_cls_nvox": ghost,                    # {c: nvox} per ghost class (summary only)
        "sdwe_ghost_nvox": int(sum(ghost.values())),     # total ghost voxels this case
        "sdwe_ghost_ncls": int(len(ghost)),              # number of ghost classes this case
        "sdwe_params": {
            "max_dist": float(max_dist),
            "dwe_norm_by": float(dwe_norm_by),
            "weight_fn": wf_name,
            "weight_margin": float(weight_margin),
            "edt_scale": DWE_SCALE,
            "edt_backend": _edt_backend(allow_gpu),
        },
    }
    if return_masks:
        res["under_seg_mask"] = under
        res["epp_mask"] = epp
    return res


# ---- self-test ------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Self-test:
        cd /mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW
        python nnunetv2/evaluation/sdwe.py
    """
    from pprint import pprint
    from nnunetv2.evaluation.distance_weighted_error import compute_distance_weighted_error

    NORM = 1.0
    MAXD = 10.0

    # ---- Test 1: under-seg depth (deep miss strictly worse than a surface shaving) ----
    gt = np.zeros((1, 11, 11), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    pred = gt.copy()
    pred[0, 3, 3] = 0                        # surface miss  -> dist-to-BG = 1
    pred[0, 5, 5] = 0                        # deep-core miss -> dist-to-BG = 3
    r = compute_simplified_dwe(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 1: under-seg depth ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert abs(r["sdwe_under_seg"] - (1.0 + 3.0)) < 1e-5, r["sdwe_under_seg"]
    assert r["sdwe_epp"] == 0.0 and r["sdwe_ghost_ncls"] == 0

    # ---- Test 2: EPP split (from-bg vs fg-conf) ----
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1                      # class 1 block (cols 3..7)
    gt[0, 3:8, 12:17] = 2                    # class 2 block (cols 12..16)
    pred = gt.copy()
    pred[0, 5, 11] = 2                       # EPP-2 on background (gt 0), adjacent to true-2 -> dist 1
    pred[0, 5, 4] = 2                        # EPP-2 on true class-1 (gt 1) -> FG<->FG confusion; dist 8
    r = compute_simplified_dwe(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 2: EPP from-bg vs fg-conf split ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert abs(r["sdwe_epp_from_bg"] - 1.0) < 1e-5, r["sdwe_epp_from_bg"]
    assert abs(r["sdwe_epp_fg_conf"] - 8.0) < 1e-5, r["sdwe_epp_fg_conf"]
    assert abs(r["sdwe_epp"] - (1.0 + 8.0)) < 1e-5
    assert r["sdwe_under_seg"] == 0.0

    # ---- Test 3: ghost class (absent from GT) punished at max_dist; counted as from-bg ----
    gt = np.zeros((1, 11, 11), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    pred = gt.copy()
    pred[0, 0, 0] = 5
    pred[0, 0, 1] = 5
    pred[0, 0, 2] = 5
    pred[0, 1, 0] = 5                        # 4 ghost voxels of class 5 (all on gt 0)
    r = compute_simplified_dwe(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 3: ghost class ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert r["sdwe_ghost_cls_nvox"] == {5: 4} and r["sdwe_ghost_nvox"] == 4 and r["sdwe_ghost_ncls"] == 1
    assert abs(r["sdwe_epp"] - 4 * MAXD) < 1e-5
    assert abs(r["sdwe_epp_from_bg"] - 4 * MAXD) < 1e-5 and r["sdwe_epp_fg_conf"] == 0.0

    # ---- Test 4: reconciliation + cross-check against the full DWE matrix scalars ----
    print("\n=== Test 4: cross-check vs distance_weighted_error ===")
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    gt[0, 3:8, 12:17] = 2
    pred = gt.copy()
    pred[0, 5, 5] = 0          # under-seg of class 1 (deep)
    pred[0, 5, 11] = 2         # epp from bg (gt 0)
    pred[0, 4, 4] = 2          # epp fg-conf (gt 1 -> pred 2)
    pred[0, 0, 0] = 7          # ghost class 7 (gt 0)
    for wfn in ('linear', 'ones', 'margin_linear', 'margin_ones'):
        s = compute_simplified_dwe(gt, pred, weight_fn=wfn, weight_margin=3.0, max_dist=MAXD, dwe_norm_by=NORM)
        d = compute_distance_weighted_error(gt, pred, weight_fn=wfn, weight_margin=3.0, max_dist=MAXD, dwe_norm_by=NORM)
        # the matrix scalars: col0 = under-seg, row0 = epp from bg, fgfg = epp fg-conf
        from nnunetv2.evaluation.distance_weighted_error import aggregate_confusion
        _, _, sc = aggregate_confusion([d['dwe_confusion']])
        assert abs(s['sdwe_under_seg'] - d['dwe_under_seg']) < 1e-9, (wfn, s, d)
        assert abs(s['sdwe_epp'] - d['dwe_epp']) < 1e-9, (wfn, s, d)
        assert abs(s['sdwe_under_seg'] - sc['dwe_cm_col0']) < 1e-9, (wfn, s, sc)
        assert abs(s['sdwe_epp_from_bg'] - sc['dwe_cm_row0']) < 1e-9, (wfn, s, sc)
        assert abs(s['sdwe_epp_fg_conf'] - sc['dwe_cm_fgfg']) < 1e-9, (wfn, s, sc)
        # 3-way reconciliation
        assert abs((s['sdwe_under_seg'] + s['sdwe_epp_from_bg'] + s['sdwe_epp_fg_conf'])
                   - (s['sdwe_under_seg'] + s['sdwe_epp'])) < 1e-9
        print(f"  {wfn:13s}: under={s['sdwe_under_seg']:.3f} epp_from_bg={s['sdwe_epp_from_bg']:.3f} "
              f"epp_fg_conf={s['sdwe_epp_fg_conf']:.3f}  == DWE col0/row0/fgfg  OK")

    print("\nAll sDWE self-tests passed.")
