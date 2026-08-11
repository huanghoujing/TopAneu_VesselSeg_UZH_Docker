"""Distance-Weighted Error (DWE) metric.

For every error voxel (gt != pred), weight it by a distance (in **voxel units**) and sum per volume:

- Under-segmentation (pred == 0, so gt == g FG): weight by distance to the nearest TRUE BACKGROUND,
  via D_bg = edt(gt != 0) indexed at the under-seg voxels. A 1-voxel surface "shaving" costs ~0; a
  miss deep in the vessel costs a lot ("severity = depth").
- Error predicted positive / EPP (pred == c, c != 0, gt != c): weight by distance to the nearest TRUE
  class-c voxel, via D_c = edt(gt != c) indexed at the EPP-c voxels. One rule absorbs both
  gt==BG->pred=c (classic false positive) and gt==other-FG->pred=c (FG<->FG confusion) -- hence "error
  predicted positive" rather than "false positive". If class c is ABSENT from GT (a "ghost" class),
  there is no distance, so each such voxel is punished with `max_dist`.

Both per-volume sums are divided by `dwe_norm_by` (default 1e6) to keep magnitudes readable.

EDT backend: cuCIM GPU (primary, bit-exact, ~5.7x faster) with a scipy CPU fallback. Because the
per-class EDT depends only on the GT, it is cached to disk (blosc2, uint16 fixed-point), keyed by GT
path + mtime, so it is reused across models / validation epochs. GPU EDT must run only in a
single-process pre-pass (`precompute_dwe_cache_for_files`); inside forked metric workers we keep
allow_gpu=False (cache reads, scipy fallback) to avoid corrupting the CUDA context.
"""
import os
from pathlib import Path

import numpy as np

from nnunetv2.evaluation.contamination_ratio_and_num_src import to_numpy, clean_numpy

# ---- fixed-point cache encoding -------------------------------------------------------------------
# Store round(min(d, DWE_MAX) * DWE_SCALE) as uint16. SCALE=16 -> 1/16-voxel resolution; the max
# storable distance ~4096 voxels is far beyond any realistic volume diagonal, so no saturation.
DWE_SCALE = 16
DWE_MAX = 65535.0 / DWE_SCALE

_MED_EXTS = ('.nii.gz', '.nrrd', '.mha', '.gipl', '.nii')


# ---- EDT backend (GPU primary, scipy fallback) ----------------------------------------------------
def _gpu_edt_available(allow_gpu=True):
    """True iff GPU EDT can run. Returns False immediately (no CUDA touch) when allow_gpu is False,
    which is what forked metric workers pass so they never initialize CUDA."""
    if not allow_gpu:
        return False
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _edt(mask_bool, allow_gpu=True):
    """Euclidean distance transform: at each non-zero/True voxel, distance to the nearest zero/False
    voxel (scipy semantics). cuCIM GPU primary, scipy CPU fallback. Returns a host float32 array."""
    mask_bool = np.ascontiguousarray(mask_bool)
    if _gpu_edt_available(allow_gpu):
        try:
            import cupy as cp
            from cucim.core.operations.morphology import distance_transform_edt as _gpu
            d = _gpu(cp.asarray(mask_bool))
            return cp.asnumpy(d).astype(np.float32)
        except Exception:
            pass
    from scipy.ndimage import distance_transform_edt as _cpu
    return _cpu(mask_bool).astype(np.float32)


def _edt_backend(allow_gpu=True):
    return 'cucim' if _gpu_edt_available(allow_gpu) else 'scipy'


# ---- uint16 fixed-point quantization --------------------------------------------------------------
def _quantize(d):
    return np.rint(np.minimum(d, DWE_MAX) * DWE_SCALE).astype(np.uint16)


def _dequantize(q):
    return q.astype(np.float32) / DWE_SCALE


# ---- blosc2 disk cache ----------------------------------------------------------------------------
def _gt_stem(gt_key):
    name = Path(str(gt_key)).name
    for ext in _MED_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _edt_cache_path(cache_dir, gt_key, c):
    """<cache_dir>/<gt_stem>/c{c}.b2nd -- one subdir per sample."""
    return Path(cache_dir) / _gt_stem(gt_key) / f"c{int(c)}.b2nd"


def _read_edt_cache(path, expect_mtime):
    """Return the decoded float32 EDT from cache, or None on miss / mtime mismatch / error."""
    try:
        import blosc2
        arr = blosc2.open(urlpath=str(path), mode='r')
        vlmeta = arr.schunk.vlmeta
        if expect_mtime is not None and 'gt_mtime' in vlmeta:
            if float(vlmeta['gt_mtime']) != float(expect_mtime):
                return None
        return _dequantize(arr[:])
    except Exception:
        return None


def _write_edt_cache(path, q_uint16, gt_mtime):
    """Write the uint16 EDT to a blosc2 .b2nd (LZ4HC/clevel5/SHUFFLE, mirroring nnUNetDataset)."""
    import blosc2
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
    cparams = {'codec': blosc2.Codec.LZ4HC, 'clevel': 5, 'nthreads': 4,
               'filters': [blosc2.Filter.SHUFFLE]}
    arr = blosc2.asarray(np.ascontiguousarray(q_uint16), urlpath=str(path), mode='w', cparams=cparams)
    try:
        arr.schunk.vlmeta['gt_mtime'] = float(gt_mtime if gt_mtime is not None else -1.0)
    except Exception:
        pass


def load_or_compute_edt(gt, c, cache_dir=None, gt_key=None, gt_mtime=None, allow_gpu=True):
    """Distance-to-nearest-class-`c` map, i.e. _edt(gt != c). Reads the blosc2 cache when
    cache_dir/gt_key are given (validating GT mtime), else computes and (best-effort) writes it.
    With no cache, computes directly and returns the full-precision float array."""
    if cache_dir is not None and gt_key is not None:
        if gt_mtime is None:
            try:
                gt_mtime = os.path.getmtime(gt_key)
            except Exception:
                gt_mtime = None
        path = _edt_cache_path(cache_dir, gt_key, c)
        if path.exists():
            cached = _read_edt_cache(path, gt_mtime)
            if cached is not None:
                return cached
        q = _quantize(_edt(gt != c, allow_gpu=allow_gpu))
        try:
            _write_edt_cache(path, q, gt_mtime)
        except Exception:
            pass
        return _dequantize(q)  # match exactly what a later cache read would return
    return _edt(gt != c, allow_gpu=allow_gpu)


def precompute_dwe_cache_for_files(files_ref, cache_dir, allow_gpu=True, verbose=True):
    """Single-process (GPU-primary) pre-pass: for each GT file, cache _edt(gt != c) for every
    c in {0} U present_fg(gt). Run this in the main process before the multiprocess metric loop so
    the workers only read the cache. Ghost (GT-absent) classes need no EDT and are skipped."""
    backend = _edt_backend(allow_gpu)
    if verbose:
        print(f"[precompute_dwe_cache_for_files] {len(files_ref)} GT files -> {cache_dir} "
              f"(edt backend: {backend})")
    for i, f in enumerate(files_ref):
        gt = to_numpy(f)
        try:
            mtime = os.path.getmtime(f)
        except Exception:
            mtime = None
        classes = [0] + [int(c) for c in np.unique(gt) if c != 0]
        for c in classes:
            path = _edt_cache_path(cache_dir, f, c)
            if path.exists() and _read_edt_cache(path, mtime) is not None:
                continue
            _write_edt_cache(path, _quantize(_edt(gt != c, allow_gpu=allow_gpu)), mtime)
        if verbose and (i + 1) % 20 == 0:
            print(f"[precompute_dwe_cache_for_files] {i + 1}/{len(files_ref)} done")


def edt_cache_to_nifti(b2nd_path, out_path=None, ref_image=None, verbose=True):
    """Decode a cached EDT `.b2nd` (uint16 fixed-point) back to a float32 distance map and write it as
    a NIFTI for a visual sanity check. The saved volume is the distance, in **voxel units**, to the
    nearest true class-`c` voxel -- `c` is encoded in the filename (`c0.b2nd` is distance-to-BG, used
    for under-seg; `c{c}.b2nd` is distance-to-true-class-c, used for EPP-c).

    Args:
        b2nd_path:  path to a `<cache_dir>/<gt_stem>/c{c}.b2nd` file.
        out_path:   output NIFTI path. Default: alongside the .b2nd as `<gt_stem>__c{c}_edt.nii.gz`.
        ref_image:  optional source GT (path or sitk image) whose spacing/origin/direction are copied
                    so the EDT overlays on the GT in a viewer. Distances stay in voxel units regardless.
    Returns the output path.
    """
    import SimpleITK as sitk
    import blosc2

    b2nd_path = Path(b2nd_path)
    arr = blosc2.open(urlpath=str(b2nd_path), mode='r')
    d = _dequantize(arr[:])                          # float32 distances, voxel units
    try:
        gt_mtime = arr.schunk.vlmeta['gt_mtime']
    except Exception:
        gt_mtime = None

    if out_path is None:
        cval = b2nd_path.stem                        # 'c0', 'c12', ...
        out_path = b2nd_path.parent / f"{b2nd_path.parent.name}__{cval}_edt.nii.gz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(d)                  # array is (z, y, x); EDT was computed on to_numpy(gt)
    if ref_image is not None:
        ref = ref_image if isinstance(ref_image, sitk.Image) else sitk.ReadImage(str(ref_image))
        if ref.GetSize() == img.GetSize():
            img.CopyInformation(ref)
        elif verbose:
            print(f"[edt_cache_to_nifti] WARNING ref size {ref.GetSize()} != edt size {img.GetSize()}; "
                  f"not copying geometry")
    sitk.WriteImage(img, str(out_path))
    if verbose:
        print(f"[edt_cache_to_nifti] {b2nd_path}  ->  {out_path}\n"
              f"  shape={d.shape} dist[min..max]={float(d.min()):.3f}..{float(d.max()):.3f} voxels"
              f"  (gt_mtime={gt_mtime})")
    return str(out_path)


def combine_edt_cache_to_nifti(sample_dir, out_path=None, t=10.0, op='min', include_c0=False,
                               classes=None, ref_image=None, as_labels=None, invert=False, verbose=True):
    """Combine all per-class EDT caches of ONE sample into a single 0~1 field for visualization.

    For each FG-class file `c{c}.b2nd` in `sample_dir` (c0 = distance-to-BG, skipped unless
    `include_c0`), the distance is clipped to `[0, t]` and the per-class fields are merged voxel-wise:
      - op='min' (default): distance to the **nearest** true class of any kind, capped at t -- a
        "near-band" field around all vessels. Rescaled `d/t` so **0 = on a vessel, 1 = at/beyond the
        t-voxel band**.
      - op='max': farthest-class distance / t (worst-case zones).
      - op='sum': sum of clipped fields, rescaled by its own max.

    Args:
        sample_dir: a `<cache_dir>/<gt_stem>/` directory holding the `c{c}.b2nd` files.
        out_path:   output NIFTI. Default `<sample_dir>/<gt_stem>__combined_t{t}_{op}.nii.gz`.
        t:          clip/threshold distance in voxels (default 10).
        classes:    optional iterable of class ids to restrict to (else all present FG classes).
        ref_image:  GT path/sitk image for geometry. Auto-detected from the cache layout
                    (`<gt_folder>/.dwe_edt_cache/<gt_stem>/`) when None.
        as_labels:  emit an INTEGER label volume (renderable in ITK-SNAP's 3D view) instead of the
                    float 0~1 field. 'shells' -> int16 `round(distance)` in 0..t (nested distance
                    shells, 0 = on a vessel); 'band' -> uint8 mask of the near band (d <= t). None
                    (default) -> the continuous float32 0~1 field (for grayscale / volume-rendering).
        invert:     flip the value scale so vessels become the HIGH/bright value: 'shells' ->
                    `max - x` (on-vessel = t), float 0~1 / 'band' -> `1 - x` (on-vessel = 1).
    Returns the output path.
    """
    import SimpleITK as sitk
    import blosc2

    sample_dir = Path(sample_dir)
    files = sorted(sample_dir.glob('c*.b2nd'), key=lambda p: int(p.stem[1:]))
    sel = []
    for p in files:
        c = int(p.stem[1:])
        if c == 0 and not include_c0:
            continue
        if classes is not None and int(c) not in set(int(x) for x in classes):
            continue
        sel.append((c, p))
    if not sel:
        raise FileNotFoundError(f"no matching c*.b2nd under {sample_dir} "
                                f"(include_c0={include_c0}, classes={classes})")

    combined = None
    for c, p in sel:
        arr = blosc2.open(urlpath=str(p), mode='r')
        d = np.minimum(_dequantize(arr[:]), float(t))      # clip distance to [0, t]
        if combined is None:
            combined = d
        elif op == 'min':
            combined = np.minimum(combined, d)
        elif op == 'max':
            combined = np.maximum(combined, d)
        elif op == 'sum':
            combined = combined + d
        else:
            raise ValueError(f"op must be 'min'/'max'/'sum', got {op!r}")

    if as_labels == 'shells':
        out_arr = np.rint(combined).astype(np.int16)        # 0..t nested distance shells
        suffix = 'shells'
    elif as_labels == 'band':
        out_arr = (combined < float(t)).astype(np.uint8)    # near-band mask (1 = within t of a vessel)
        suffix = 'band'
    elif as_labels is None:
        denom = float(t) if op in ('min', 'max') else float(combined.max()) or 1.0
        out_arr = (combined / denom).astype(np.float32)     # -> 0~1
        suffix = op
    else:
        raise ValueError(f"as_labels must be None / 'shells' / 'band', got {as_labels!r}")

    if invert:                                              # make vessels the HIGH/bright value
        if as_labels == 'shells':
            out_arr = (out_arr.max() - out_arr).astype(out_arr.dtype)   # max - x
        else:                                               # float 0~1 or band mask
            out_arr = (1 - out_arr).astype(out_arr.dtype)               # 1 - x
        suffix += '_inv'

    # auto-detect the source GT for geometry: <gt_folder>/.dwe_edt_cache/<gt_stem>/
    if ref_image is None:
        gt_folder = sample_dir.parent.parent
        for ext in _MED_EXTS:
            cand = gt_folder / f"{sample_dir.name}{ext}"
            if cand.exists():
                ref_image = cand
                break

    if out_path is None:
        out_path = sample_dir / f"{sample_dir.name}__combined_t{float(t):g}_{suffix}.nii.gz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(out_arr)
    if ref_image is not None:
        ref = ref_image if isinstance(ref_image, sitk.Image) else sitk.ReadImage(str(ref_image))
        if ref.GetSize() == img.GetSize():
            img.CopyInformation(ref)
        elif verbose:
            print(f"[combine_edt_cache_to_nifti] WARNING ref size {ref.GetSize()} != "
                  f"{img.GetSize()}; not copying geometry")
    sitk.WriteImage(img, str(out_path))
    if verbose:
        print(f"[combine_edt_cache_to_nifti] {len(sel)} classes {sorted(c for c, _ in sel)} "
              f"op={op} t={t} out={suffix}({out_arr.dtype})  ->  {out_path}\n"
              f"  shape={out_arr.shape} value[min..max]={float(out_arr.min()):.3f}.."
              f"{float(out_arr.max()):.3f}  (ref={'auto:'+str(ref_image) if ref_image is not None else None})")
    return str(out_path)


# ---- weight functions -----------------------------------------------------------------------------
# Module-level so they pickle across the spawn worker pool. Through the pipeline a weight_fn is passed
# BY NAME (a string) + a margin and resolved here inside the worker, so a (possibly closure) callable
# is never pickled. The same weight_fn governs BOTH the scalar sums and the confusion matrix, so
# weight_fn='ones' yields an error-voxel-COUNT matrix and 'margin_linear' forgives near-boundary errors.
def _wf_linear(d):
    return np.asarray(d, dtype=np.float64)            # w = d


def _wf_ones(d):
    return np.ones(np.shape(d), dtype=np.float64)     # w = 1  -> voxel counts


def make_margin_weight_fn(margin, base='linear'):
    """Forgive near-boundary errors (w = 0 for d <= margin), then apply `base` above the margin:
    base='linear' -> w = d  ('margin_linear');  base='ones' -> w = 1  ('margin_ones', a thresholded count)."""
    margin = float(margin)
    if base == 'ones':
        def _wf(d):
            d = np.asarray(d, dtype=np.float64)
            return np.where(d <= margin, 0.0, 1.0)
        _wf.__name__ = f'margin{margin:g}_ones'
    else:  # 'linear'
        def _wf(d):
            d = np.asarray(d, dtype=np.float64)
            return np.where(d <= margin, 0.0, d)
        _wf.__name__ = f'margin{margin:g}_linear'
    return _wf


# Names accepted by resolve_weight_fn (and the --dwe_weight_fn CLI choices).
WEIGHT_FN_NAMES = ('linear', 'ones', 'margin_linear', 'margin_ones')


def resolve_weight_fn(weight_fn=None, weight_margin=3.0):
    """Return (callable, name). `weight_fn` may be a callable or one of WEIGHT_FN_NAMES:
    'linear' (w=d), 'ones' (w=1, counts), 'margin_linear' (0 if d<=margin else d),
    'margin_ones' (0 if d<=margin else 1, a margin-thresholded count)."""
    if weight_fn is None or weight_fn == 'linear':
        return _wf_linear, 'linear'
    if weight_fn == 'ones':
        return _wf_ones, 'ones'
    if weight_fn == 'margin_linear':
        fn = make_margin_weight_fn(weight_margin, 'linear')
        return fn, fn.__name__
    if weight_fn == 'margin_ones':
        fn = make_margin_weight_fn(weight_margin, 'ones')
        return fn, fn.__name__
    if callable(weight_fn):
        return weight_fn, getattr(weight_fn, '__name__', repr(weight_fn))
    raise ValueError(f"Unknown weight_fn {weight_fn!r}; expected a callable or one of {WEIGHT_FN_NAMES}")


def weight_fn_is_distance_free(weight_fn):
    """True iff the weighting ignores the distance value (only 'ones'), so the EDT — and the whole
    GPU pre-pass / cache — can be skipped entirely. Distance-dependent fns (incl. margin variants and
    arbitrary callables) return False."""
    return weight_fn == 'ones' or weight_fn is _wf_ones


# ---- the metric -----------------------------------------------------------------------------------
def compute_distance_weighted_error(gt, pred, weight_fn=None, weight_margin=3.0, max_dist=500.0,
                                    dwe_norm_by=1e6, edt_cache_dir=None, gt_key=None, gt_mtime=None,
                                    allow_gpu=True, return_masks=False):
    """Distance-Weighted Error for one (gt, pred) pair. See module docstring.

    weight_fn:   None/'linear'/'ones'/'margin_linear'/'margin_ones' or a callable (see resolve_weight_fn). Governs both
                 the scalar sums and the confusion matrix.
    weight_margin: margin (voxels) for weight_fn='margin_linear'/'margin_ones'.
    max_dist:    punishment distance (voxels) for EPP voxels of a GT-absent ("ghost") class.
    dwe_norm_by: divisor applied to every reported sum / matrix cell.
    edt_cache_dir / gt_key / gt_mtime: enable the blosc2 EDT cache (gt_key is the GT file path).
    allow_gpu:   keep False inside forked workers (cache read + scipy fallback only).

    The per-case confusion matrix `dwe_confusion` is {gt: {pred: scaled weighted mass}}: rows are the
    present GT classes (incl 0, so an empty row marks "present with zero errors" -> the NaN mask for
    dataset-level averaging), columns are the error pred classes (incl 0 = under-seg, incl ghost).
    Sum over all its cells == dwe_under_seg + dwe_epp.
    """
    gt = to_numpy(gt)
    pred = to_numpy(pred)
    wf, wf_name = resolve_weight_fn(weight_fn, weight_margin)

    error = gt != pred
    under = error & (pred == 0)               # under-segmentation (matrix column p == 0)
    epp = error & (pred != 0)                 # error predicted positive (columns p != 0)

    # When the weighting ignores distance (weight_fn='ones'), the EDT values are never used, so we
    # skip every EDT compute/load (and the caller skips the GPU pre-pass): just feed a dummy zero
    # array of the right length to wf, which returns the distance-independent weights.
    distance_free = weight_fn_is_distance_free(weight_fn)

    def _edt_c(c):
        return load_or_compute_edt(gt, c, edt_cache_dir, gt_key, gt_mtime, allow_gpu)

    conf = {int(g): {} for g in np.unique(gt)}      # rows = present GT classes (incl 0)

    def _accumulate(sel_mask, w, col):
        """Bin per-voxel weights `w` (for voxels `sel_mask`, predicted as `col`) by GT class into the
        confusion matrix; return the scaled total."""
        gv = gt[sel_mask]
        for g in np.unique(gv):
            s = float(w[gv == g].sum()) / dwe_norm_by
            if s:
                conf[int(g)][col] = conf[int(g)].get(col, 0.0) + s
        return float(w.sum()) / dwe_norm_by

    # under-seg -> column 0
    if under.any():
        wu = wf(np.zeros(int(under.sum()))) if distance_free else wf(_edt_c(0)[under])
        sum_under = _accumulate(under, wu, 0)
    else:
        sum_under = 0.0

    # epp -> column c
    sum_epp = 0.0
    sum_epp_from_bg = 0.0                                 # EPP where gt == 0 (FP from background) = row0
    sum_epp_fg_conf = 0.0                                 # EPP where gt != 0 (FG<->FG confusion)  = fgfg
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
        s_c = _accumulate(sel, w, c)
        sum_epp += s_c
        per_cls[c] = s_c
        from_bg = gt[sel] == 0                            # split this class's EPP mass by true label
        sum_epp_from_bg += float(w[from_bg].sum()) / dwe_norm_by
        sum_epp_fg_conf += float(w[~from_bg].sum()) / dwe_norm_by

    res = {
        "dwe_under_seg": sum_under,
        "dwe_epp": sum_epp,
        "dwe_error": sum_under + sum_epp,                # total error mass (== sDWE/ssDWE total)
        "dwe_epp_from_bg": sum_epp_from_bg,              # FP-from-background mass (== matrix row0)
        "dwe_epp_fg_conf": sum_epp_fg_conf,              # FG<->FG confusion mass (== matrix fgfg)
        "dwe_epp_per_pred_cls": per_cls,                 # scaled, debug (summary only)
        "dwe_ghost_cls_nvox": ghost,                     # {c: nvox} per ghost class (summary only)
        "dwe_ghost_nvox": int(sum(ghost.values())),      # total ghost voxels this case
        "dwe_ghost_ncls": int(len(ghost)),               # number of ghost classes this case
        "dwe_confusion": conf,                           # {gt: {pred: scaled mass}}, errors only
        "dwe_params": {
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


# ---- dataset-level confusion aggregation + export -------------------------------------------------
def aggregate_confusion(confs):
    """confs: list of per-case {gt: {pred: mass}} dicts (rows = present GT classes that case).

    Returns (M, classes, scalars):
      M       : {g: {p: nanmean over cases where g is present}} (NaN where g never present anywhere)
      classes : sorted union of all row/col classes seen
      scalars : dwe_cm_sum / dwe_cm_row0 (BG->FG = false positives) / dwe_cm_col0 (under-seg) /
                dwe_cm_fgfg (FG<->FG confusion); nansum semantics, and sum == row0 + col0 + fgfg.
    """
    def _int_keys(d):
        return {int(k): {int(kk): float(vv) for kk, vv in v.items()} for k, v in d.items()}
    confs = [_int_keys(cf) for cf in confs]

    classes = set()
    for cf in confs:
        for g, row in cf.items():
            classes.add(g)
            classes.update(row.keys())
    classes = sorted(classes)

    M = {}
    for g in classes:
        M[g] = {}
        for p in classes:
            vals = [cf[g].get(p, 0.0) for cf in confs if g in cf]   # g in cf == row active (present)
            M[g][p] = float(np.mean(vals)) if vals else float('nan')

    def _nansum(xs):
        return float(np.nansum(list(xs)))
    row0 = _nansum(M[0][p] for p in classes) if 0 in classes else 0.0
    col0 = _nansum(M[g][0] for g in classes) if 0 in classes else 0.0
    fgfg = _nansum(M[g][p] for g in classes for p in classes if g != 0 and p != 0)
    total = _nansum(M[g][p] for g in classes for p in classes)
    scalars = {'dwe_cm_sum': total, 'dwe_cm_row0': row0, 'dwe_cm_col0': col0, 'dwe_cm_fgfg': fgfg}
    return M, classes, scalars


def confusion_to_dense(M, classes):
    """{g:{p:v}} -> dense (len x len) float array (NaN where missing), aligned to `classes`. Keys may
    be str or int; a value of None (e.g. a JSON `null`) is read as NaN."""
    A = np.full((len(classes), len(classes)), np.nan, dtype=np.float64)
    idx = {c: i for i, c in enumerate(classes)}
    for g, row in M.items():
        for p, v in row.items():
            A[idx[int(g)], idx[int(p)]] = np.nan if v is None else float(v)
    return A


def save_confusion_matrix_png(A, classes, out_path, scale='log', title=None, cmap='gray',
                              vmin=None, vmax=None):
    """Heatmap of the dataset confusion matrix. scale='log' (default) or 'linear'. Grayscale by
    default with NaN (and, under log, non-positive) cells black -- so on the 'gray' colormap a zero and
    an absent (NaN) cell both read black and brighter = larger error. A faint grid marks cell borders
    for easy row/col index reading. Integer class indices on both axes.

    vmin/vmax: pin the color scale to a constant so PNGs from DIFFERENT models are visually
    comparable (cells map to the same brightness everywhere). When None (default) each axis is derived
    from THIS matrix (log: min/max of positive cells; linear: vmin=0, vmax=max cell) -- handy for a
    single matrix but NOT comparable across models. For log scale vmin is floored at 1e-12."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    A = np.array(A, dtype=np.float64)
    if scale == 'log':
        data = np.ma.masked_where(~np.isfinite(A) | (A <= 0), A)
        pos = data.compressed()
        _vmin = vmin if vmin is not None else (float(pos.min()) if pos.size else 1e-6)
        _vmax = vmax if vmax is not None else (float(pos.max()) if pos.size else 1.0)
        norm = LogNorm(vmin=max(_vmin, 1e-12), vmax=max(_vmax, _vmin * 10))
    else:
        data = np.ma.masked_invalid(A)
        _vmin = vmin if vmin is not None else 0.0
        _vmax = vmax if vmax is not None else (float(data.max()) if data.count() else 1.0)
        norm = Normalize(vmin=_vmin, vmax=_vmax if _vmax > _vmin else _vmin + 1.0)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('black')                                    # NaN (and masked non-positive under log) -> black
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(5, n * 0.35)))
    im = ax.imshow(data, cmap=cm, norm=norm, aspect='equal')
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=6)
    # faint grid on cell borders (minor ticks at -0.5 .. n-0.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='0.6', linewidth=0.4, alpha=0.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('GT class')
    ax.set_title(title or f'DWE confusion ({scale} scale)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---- redraw PNGs from saved artifacts (no re-evaluation) ------------------------------------------
def load_confusion_from_summary(summary_json_path):
    """Load the dataset confusion matrix from a saved summary.json (overall_dwe.dwe_confusion).
    Returns (classes, dense). NaN cells (absent-GT rows) are preserved."""
    import json
    with open(summary_json_path) as f:
        d = json.load(f)                       # stdlib json reads the `NaN` literal back as float nan
    M = d['overall_dwe']['dwe_confusion']      # {str(g): {str(p): val|null}}
    classes = sorted({int(k) for k in M} | {int(p) for row in M.values() for p in row})
    return classes, confusion_to_dense(M, classes)


def load_confusion_from_csv(csv_path):
    """Load the dataset confusion matrix from a saved dwe_confusion_matrix.csv
    ('GT Label' + 'Pred <c>' columns). Returns (classes, dense); empty cells -> NaN."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    row_classes = [int(x) for x in df['GT Label'].tolist()]
    pred_cols = [c for c in df.columns if str(c).startswith('Pred ')]
    col_classes = [int(str(c)[len('Pred '):]) for c in pred_cols]
    classes = sorted(set(row_classes) | set(col_classes))
    vals = df[pred_cols].to_numpy(dtype=np.float64)   # pandas reads blanks as NaN
    M = {g: {p: vals[i, j] for j, p in enumerate(col_classes)}
         for i, g in enumerate(row_classes)}
    return classes, confusion_to_dense(M, classes)


def redraw_confusion_png(source, out_dir=None, prefix='dwe_confusion_matrix',
                         scales=('log', 'linear'), vmin=None, vmax=None, verbose=True):
    """Regenerate the DWE confusion heatmaps from a saved summary.json OR dwe_confusion_matrix.csv,
    without re-running evaluation. out_dir defaults to the source file's directory. vmin/vmax pin the
    color scale to a constant (pass the same values when redrawing several models to compare them)."""
    source = str(source)
    if source.endswith('.json'):
        classes, A = load_confusion_from_summary(source)
    elif source.endswith('.csv'):
        classes, A = load_confusion_from_csv(source)
    else:
        raise ValueError(f"source must be a .json (summary) or .csv (confusion matrix); got {source!r}")
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(source))
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for scale in scales:
        out_path = os.path.join(out_dir, f"{prefix}_{scale}.png")
        save_confusion_matrix_png(A, classes, out_path, scale=scale, vmin=vmin, vmax=vmax)
        written.append(out_path)
        if verbose:
            print(f"[redraw_confusion_png] wrote {out_path}")
    return written


# ---- self-test ------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Self-test:
        cd /mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW
        python nnunetv2/evaluation/distance_weighted_error.py
    Redraw the confusion PNGs from a saved summary.json or dwe_confusion_matrix.csv (no re-eval):
        python nnunetv2/evaluation/distance_weighted_error.py --redraw <summary.json|matrix.csv> [--out_dir DIR]
    Decode a cached EDT .b2nd to a NIFTI for a visual sanity check:
        python nnunetv2/evaluation/distance_weighted_error.py --to_nifti <c{c}.b2nd> [--ref GT.nii.gz] [--out OUT.nii.gz]
    Combine one sample's per-class EDT caches into a single near-band volume (float 0~1, or an integer
    label volume renderable in ITK-SNAP's 3D view via --as_labels shells|band):
        python nnunetv2/evaluation/distance_weighted_error.py --combine <sample_dir> [--t 10] [--op min] [--include_c0] [--as_labels shells|band] [--invert] [--ref GT.nii.gz] [--out OUT.nii.gz]
    """
    import sys
    if '--redraw' in sys.argv:
        src = sys.argv[sys.argv.index('--redraw') + 1]
        out_dir = sys.argv[sys.argv.index('--out_dir') + 1] if '--out_dir' in sys.argv else None
        redraw_confusion_png(src, out_dir=out_dir)
        sys.exit(0)

    if '--to_nifti' in sys.argv:
        src = sys.argv[sys.argv.index('--to_nifti') + 1]
        ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
        out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
        edt_cache_to_nifti(src, out_path=out, ref_image=ref)
        sys.exit(0)

    if '--combine' in sys.argv:
        sample_dir = sys.argv[sys.argv.index('--combine') + 1]
        t = float(sys.argv[sys.argv.index('--t') + 1]) if '--t' in sys.argv else 10.0
        op = sys.argv[sys.argv.index('--op') + 1] if '--op' in sys.argv else 'min'
        include_c0 = '--include_c0' in sys.argv
        ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
        out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
        as_labels = sys.argv[sys.argv.index('--as_labels') + 1] if '--as_labels' in sys.argv else None
        invert = '--invert' in sys.argv
        combine_edt_cache_to_nifti(sample_dir, out_path=out, t=t, op=op, include_c0=include_c0,
                                   ref_image=ref, as_labels=as_labels, invert=invert)
        sys.exit(0)

    import tempfile
    from pprint import pprint

    NORM = 1.0          # no scaling for exact arithmetic in the tests
    MAXD = 10.0         # small, readable ghost punishment for the tests

    # ---- Test 1: under-seg depth (deep miss strictly worse than a surface shaving) ----
    gt = np.zeros((1, 11, 11), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1                      # 5x5 FG block of class 1
    pred = gt.copy()
    pred[0, 3, 3] = 0                        # surface miss  -> dist-to-BG = 1
    pred[0, 5, 5] = 0                        # deep-core miss -> dist-to-BG = 3
    r = compute_distance_weighted_error(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 1: under-seg depth ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert abs(r["dwe_under_seg"] - (1.0 + 3.0)) < 1e-5, r["dwe_under_seg"]
    assert r["dwe_epp"] == 0.0 and r["dwe_ghost_ncls"] == 0

    # ---- Test 2: EPP weighted by distance to the predicted class (far worse than near) ----
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1                      # class 1 block (cols 3..7)
    gt[0, 3:8, 12:17] = 2                    # class 2 block (cols 12..16)
    pred = gt.copy()
    pred[0, 5, 11] = 2                       # EPP-2 adjacent to true-2 -> dist 1
    pred[0, 5, 0] = 2                        # EPP-2 far from true-2    -> dist 12
    r = compute_distance_weighted_error(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 2: EPP distance to predicted class ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert abs(r["dwe_epp"] - (1.0 + 12.0)) < 1e-5, r["dwe_epp"]
    assert r["dwe_under_seg"] == 0.0 and r["dwe_ghost_ncls"] == 0

    # ---- Test 3: ghost class (absent from GT) punished at max_dist, counted in diagnostics ----
    gt = np.zeros((1, 11, 11), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    pred = gt.copy()
    pred[0, 0, 0] = 5                        # class 5 absent from GT
    pred[0, 0, 1] = 5
    pred[0, 0, 2] = 5
    pred[0, 1, 0] = 5                        # 4 ghost voxels of class 5
    r = compute_distance_weighted_error(gt, pred, max_dist=MAXD, dwe_norm_by=NORM)
    print("\n=== Test 3: ghost class ===")
    pprint(clean_numpy({k: v for k, v in r.items() if not k.endswith('_mask')}))
    assert r["dwe_ghost_cls_nvox"] == {5: 4}, r["dwe_ghost_cls_nvox"]
    assert r["dwe_ghost_nvox"] == 4 and r["dwe_ghost_ncls"] == 1
    assert abs(r["dwe_epp"] - 4 * MAXD) < 1e-5, r["dwe_epp"]

    # ---- Test 4: GPU/scipy backend equivalence + uint16 quantization roundtrip ----
    print("\n=== Test 4: backend + quantization ===")
    big = np.zeros((96, 96, 64), dtype=bool)
    big[40:60, 30:50, 10:40] = True
    mask = ~big                              # distance to the blob
    d_cpu = _edt(mask, allow_gpu=False)
    d_gpu = _edt(mask, allow_gpu=True)
    print(f"edt backend (allow_gpu=True): {_edt_backend(True)};  "
          f"max|gpu-cpu| = {np.abs(d_gpu - d_cpu).max():.6f}")
    assert np.abs(d_gpu - d_cpu).max() < 1e-4
    rt = _dequantize(_quantize(d_cpu))
    assert np.abs(rt - d_cpu).max() <= 1.0 / DWE_SCALE + 1e-6

    # ---- Test 5: disk cache write/read roundtrip + mtime invalidation ----
    print("\n=== Test 5: blosc2 cache ===")
    with tempfile.TemporaryDirectory() as tmp:
        gt = np.zeros((1, 11, 11), dtype=np.int16)
        gt[0, 3:8, 3:8] = 1
        gt_path = os.path.join(tmp, "case_GT.nii.gz")
        import SimpleITK as sitk
        sitk.WriteImage(sitk.GetImageFromArray(gt), gt_path)
        mtime = os.path.getmtime(gt_path)
        cache = os.path.join(tmp, "cache")
        d1 = load_or_compute_edt(gt, 0, cache, gt_path, mtime, allow_gpu=True)   # compute + write
        assert _edt_cache_path(cache, gt_path, 0).exists()
        d2 = load_or_compute_edt(gt, 0, cache, gt_path, mtime, allow_gpu=False)  # read from cache
        assert np.array_equal(d1, d2), "cache read must match the written (quantized) array"
        d3 = load_or_compute_edt(gt, 0, cache, gt_path, mtime + 123.0, allow_gpu=False)  # mtime miss
        assert d3 is not None
        print(f"cache file: {_edt_cache_path(cache, gt_path, 0)} ; roundtrip max|d1-d2| = "
              f"{np.abs(d1 - d2).max():.6f}")

    # ---- Test 6: confusion matrix (reconciliation, weight_fn variants, dataset aggregation) ----
    print("\n=== Test 6: confusion matrix ===")
    gt = np.zeros((1, 11, 17), dtype=np.int16)
    gt[0, 3:8, 3:8] = 1
    gt[0, 3:8, 12:17] = 2
    pred = gt.copy()
    pred[0, 5, 5] = 0          # under-seg of class 1 (deep)  -> cell[1][0] = 3
    pred[0, 5, 11] = 2         # epp near true-2 (gt 0)        -> cell[0][2] = 1
    pred[0, 0, 0] = 7          # ghost class 7 (gt 0)          -> cell[0][7] = max_dist

    def _cellsum(conf):
        return sum(v for row in conf.values() for v in row.values())

    r = compute_distance_weighted_error(gt, pred, weight_fn='linear', max_dist=10.0, dwe_norm_by=1.0)
    pprint(clean_numpy(r['dwe_confusion']))
    assert r['dwe_confusion'] == {0: {2: 1.0, 7: 10.0}, 1: {0: 3.0}, 2: {}}, r['dwe_confusion']
    assert abs(_cellsum(r['dwe_confusion']) - (r['dwe_under_seg'] + r['dwe_epp'])) < 1e-6  # reconcile
    # the flat scalars mirror the matrix decomposition: error == under+epp; from_bg == row0; fg_conf == fgfg
    _, _, _sc = aggregate_confusion([r['dwe_confusion']])
    assert abs(r['dwe_error'] - (r['dwe_under_seg'] + r['dwe_epp'])) < 1e-9
    assert abs(r['dwe_epp_from_bg'] - _sc['dwe_cm_row0']) < 1e-9
    assert abs(r['dwe_epp_fg_conf'] - _sc['dwe_cm_fgfg']) < 1e-9
    assert abs(r['dwe_under_seg'] - _sc['dwe_cm_col0']) < 1e-9
    assert abs(r['dwe_epp'] - (r['dwe_epp_from_bg'] + r['dwe_epp_fg_conf'])) < 1e-9

    rc = compute_distance_weighted_error(gt, pred, weight_fn='ones', max_dist=10.0, dwe_norm_by=1.0)
    assert rc['dwe_confusion'] == {0: {2: 1.0, 7: 1.0}, 1: {0: 1.0}, 2: {}}, rc['dwe_confusion']  # counts

    rm = compute_distance_weighted_error(gt, pred, weight_fn='margin_linear', weight_margin=3.0,
                                         max_dist=10.0, dwe_norm_by=1.0)
    # margin=3: under-seg d=3 -> 0, epp d=1 -> 0; only the far ghost (max_dist 10 > 3) survives
    assert rm['dwe_confusion'] == {0: {7: 10.0}, 1: {}, 2: {}}, rm['dwe_confusion']

    confA = {0: {2: 1.0}, 1: {0: 3.0}, 2: {}}
    confB = {0: {7: 10.0}, 1: {0: 5.0}}          # class 2 absent in case B
    M, classes, sc = aggregate_confusion([confA, confB])
    print("classes:", classes, "scalars:", clean_numpy(sc))
    assert classes == [0, 1, 2, 7]
    assert M[0][2] == 0.5 and M[0][7] == 5.0 and M[1][0] == 4.0
    assert M[7][0] != M[7][0]                      # row 7 is pure-ghost -> NaN
    assert abs(sc['dwe_cm_sum'] - 9.5) < 1e-9
    assert abs(sc['dwe_cm_sum'] - (sc['dwe_cm_row0'] + sc['dwe_cm_col0'] + sc['dwe_cm_fgfg'])) < 1e-9
    A = confusion_to_dense(M, classes)
    with tempfile.TemporaryDirectory() as tmp:
        for scale in ('log', 'linear'):
            png = os.path.join(tmp, f'cm_{scale}.png')
            save_confusion_matrix_png(A, classes, png, scale=scale)
            assert os.path.getsize(png) > 0
        print("PNG (log/linear) written OK")

    print("\nAll DWE self-tests passed.")
