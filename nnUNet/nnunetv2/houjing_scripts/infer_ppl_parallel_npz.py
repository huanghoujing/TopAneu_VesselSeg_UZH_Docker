"""
Use pipeline parallel to speed up inference on test set.
Pipeline Phases:
    1. Image preprocessing, saving to queue 1 (writes tmp .npz files)
    2. GPU inference & resample & fusion, reading from queue 1, saving to queue 2
    3. Post-processing and saving segmentations to disk, reading from queue 2
User can set queue size, and number of workers for each phase.

The .npz mechamism instead of passing preprocessed data directly via queue solves the error happening from time to time:

    Traceback (most recent call last): 
    File "/mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW/nnunetv2/houjing_scripts/infer_ppl_parallel.py", 
        line 135, in inference_worker item = queue1.get() 
    File "/home/x/miniconda3/lib/python3.13/multiprocessing/queues.py", 
        line 120, in get return _ForkingPickler.loads(res) ~~~~~~~~~~~~~~~~~~~~~^^^^^ 
    File "/home/x/miniconda3/lib/python3.13/site-packages/torch/multiprocessing/reductions.py", 
        line 541, in rebuild_storage_fd fd = df.detach() 
    File "/home/x/miniconda3/lib/python3.13/multiprocessing/resource_sharer.py", 
        line 57, in detach with _resource_sharer.get_connection(self._id) as conn: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^ 
    File "/home/x/miniconda3/lib/python3.13/multiprocessing/resource_sharer.py", 
        line 86, in get_connection c = Client(address, authkey=process.current_process().authkey) 
    File "/home/x/miniconda3/lib/python3.13/multiprocessing/connection.py", 
        line 519, in Client c = SocketClient(address) 
    File "/home/x/miniconda3/lib/python3.13/multiprocessing/connection.py", 
        line 647, in SocketClient s.connect(address) ~~~~~~~~~^^^^^^^^^
    FileNotFoundError: [Errno 2] No such file or directory
"""
from shlex import join
import hashlib
import torch
import os
import sys
from pathlib import Path
import json
from glob import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import nibabel as nib
import skimage
import time
from loguru import logger
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import random
import scipy.ndimage
from functools import partial

# Ensure multiprocess start method and torch sharing strategy BEFORE any Process/Queue creation
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # already set in this interpreter
    pass

try:
    import torch.multiprocessing as _tmp_mp
    try:
        _tmp_mp.set_sharing_strategy("file_system")
    except Exception:
        pass
except Exception:
    pass

# This controls which nnUNet code to use
# If not set, the one in the PYTHONPATH will be used
# sys.path.insert(0, '/home/houjing/Project/nnUNet')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.postprocessing.remove_connected_components import remove_all_but_largest_component_from_segmentation
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, generic_filter_components
# from skimage.morphology import binary_dilation, binary_erosion
from nnunetv2.utilities.file_path_utilities import subfiles
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_prob_with_correct_shape
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
import torch.nn.functional as F

def get_predictors(model_cfg, predictor_class, use_mirroring, device_id=0):
    predictors = []
    for m_cfg in model_cfg:
        base_model_dir = m_cfg['base_model_dir']
        predictor = predictor_class(
            tile_step_size=m_cfg.get('tile_step_size', None) or 0.5,  # Default tile step size is 0.5
            use_gaussian=True,
            use_mirroring=use_mirroring,  # If True, may use mirroring augmentation, according to the plans file
            device=torch.device('cuda', device_id),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        subdir = m_cfg['subdir']
        model_fold = m_cfg['model_fold']
        checkpoint_name = m_cfg['ckpt_name']
        predictor.initialize_from_trained_model_folder(f"{base_model_dir}/{subdir}", use_folds=(model_fold,), checkpoint_name=checkpoint_name)
        logger.info(f'[{subdir}] Original Patch Size: {predictor.configuration_manager.patch_size}')
        if m_cfg.get('patch_size') is not None:
            predictor.configuration_manager.configuration["patch_size"] = m_cfg['patch_size']
        logger.info(f"[{subdir}] patch size: {predictor.configuration_manager.patch_size}")
        logger.info(f"[{subdir}] tile_step_size: {predictor.tile_step_size}")
        predictors.append(predictor)
    logger.info(f"[Predictors] Creation Done")
    return predictors

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected for argument, got {value}")

# TODO: TMP: Remove this
def prob_to_seg(prob, *args, **kwargs):
    # return (prob[1] > 0.2).astype(np.uint8)
    return np.argmax(prob, axis=0).astype(np.uint8)

def _resample_slab_trilinear(src, z0, z1, d_out, hw_out):
    """Trilinear-resample target-grid z-slab [z0, z1) from the full source volume src (C, d, h, w).

    Reproduces F.interpolate(src, (d_out, *hw_out), mode='trilinear', align_corners=False,
    antialias=False) restricted to the slab, via the separable 1D-z lerp + 2D bilinear
    decomposition (exact up to float rounding order). Computes in float32 regardless of
    src dtype. Returns (C, z1-z0, *hw_out) float32.
    """
    C, d_in, h_in, w_in = src.shape
    if d_out == d_in:
        lerped = src[:, z0:z1].float()
    else:
        # align_corners=False mapping: src_coord = (dst + 0.5) * (in / out) - 0.5
        zc = (torch.arange(z0, z1, dtype=torch.float64) + 0.5) * (d_in / d_out) - 0.5
        zf = torch.floor(zc)
        w = (zc - zf).float().view(1, -1, 1, 1)
        i0 = zf.long().clamp_(0, d_in - 1)
        i1 = (zf.long() + 1).clamp_(0, d_in - 1)
        lerped = src[:, i0].float() * (1 - w) + src[:, i1].float() * w
    if (h_in, w_in) == tuple(hw_out):
        return lerped.contiguous()
    nz = lerped.shape[1]
    return F.interpolate(lerped.reshape(1, C * nz, h_in, w_in), size=tuple(hw_out),
                         mode='bilinear', antialias=False).reshape(C, nz, *hw_out)


# z-slab thickness is chosen so one full-resolution slab buffer stays below this
_SLAB_BUDGET_BYTES = 256 * 1024 ** 2

def _streamed_seg_from_sources(sources, n_sources, pm, cm, lm, props, fp16=False):
    """Fuse logits sources into a segmentation without materializing any full-resolution
    float volume per source.

    sources yields n_sources logits tensors (C, d, h, w) on the shared preprocessed grid
    (consumed lazily, one resident at a time). Each source is streamed in z-slabs through
    resample + softmax into a running sum (n_sources > 1), or argmaxed directly per slab
    (n_sources == 1, softmax skipped: it is monotone per voxel so the argmax is unchanged).
    The sum is not divided by n_sources before the argmax for the same reason.
    Returns the segmentation as uint8 in the original array layout.
    """
    new_shape = [int(i) for i in props['shape_after_cropping_and_before_resampling']]
    d_out, h_out, w_out = new_shape
    acc = None
    seg_cropped = np.empty(new_shape, dtype=np.uint8)
    for src in sources:
        C = src.shape[0]
        nz = min(d_out, max(1, _SLAB_BUDGET_BYTES // (C * h_out * w_out * 4)))
        if n_sources > 1 and acc is None:
            acc = torch.zeros((C, *new_shape), dtype=torch.float16 if fp16 else torch.float32)
        for z0 in range(0, d_out, nz):
            z1 = min(z0 + nz, d_out)
            slab = _resample_slab_trilinear(src, z0, z1, d_out, (h_out, w_out))
            if n_sources == 1:
                seg_cropped[z0:z1] = torch.argmax(slab, dim=0).numpy().astype(np.uint8)
            else:
                acc[:, z0:z1] += lm.apply_inference_nonlin(slab).to(acc.dtype)
            del slab
        del src
    if acc is not None:
        for z0 in range(0, d_out, nz):
            z1 = min(z0 + nz, d_out)
            seg_cropped[z0:z1] = torch.argmax(acc[:, z0:z1], dim=0).numpy().astype(np.uint8)
        del acc

    # revert cropping: outside the bbox the reference pipeline pads background prob 1
    # (revert_cropping_on_probabilities), whose argmax is label 0
    seg_full = np.zeros([int(i) for i in props['shape_before_cropping']], dtype=np.uint8)
    seg_full[tuple(slice(b[0], b[1]) for b in props['bbox_used_for_cropping'])] = seg_cropped
    return seg_full.transpose(pm.transpose_backward)


@torch.inference_mode()
def predict_and_fuse(predictors, preprocessed_dicts, plans_managers, configuration_managers, label_managers, fuse_logits=False, fp16=False, streamed=True, verbose=False):
    """Predict all models on one preprocessed case and fuse into a segmentation (uint8,
    original array layout). Returns None if there is nothing to predict.

    fuse_logits=False: per model resample logits + softmax, then arithmetic mean of the
    full-resolution probability maps (the original ensemble semantics).
    fuse_logits=True: mean of the logits on the preprocessed grid, then a single
    resample (+ softmax). Requires all models to share the same preprocessing geometry
    (target spacing, transpose, cropping); fusion semantics change from mean-of-softmax
    to softmax-of-mean-logits. fp16 additionally keeps the fused logits in float16.

    streamed=True (default) processes the resample/softmax/argmax in z-slabs so no
    full-resolution float volume is materialized per model (with fuse_logits, none at
    all); output is equivalent to the full-volume path up to float rounding order.
    Falls back to the full-volume path when the plans would use separate-z resampling
    or region-based labels, where the slab decomposition does not apply.
    """
    n = min(len(predictors), len(preprocessed_dicts))
    if n == 0:
        return None
    pm, cm, lm = plans_managers[0], configuration_managers[0], label_managers[0]
    props = preprocessed_dicts[0].get('data_properties', {})

    if fuse_logits:
        # logits can only be averaged on a shared grid; identical shapes alone are not
        # proof (different spacings can round to the same shape), so check the plans
        spacings = [tuple(c.spacing) for c in configuration_managers[:n]]
        transposes = [tuple(p.transpose_backward) for p in plans_managers[:n]]
        if len(set(spacings)) > 1 or len(set(transposes)) > 1:
            raise ValueError(
                f"fuse_logits requires all models to share the same preprocessing target "
                f"spacing and transpose; got spacings={spacings}, transposes={transposes}")

    if streamed:
        try:
            do_sep = False
            for _cm in configuration_managers[:n]:
                current_spacing = _cm.spacing if \
                    len(_cm.spacing) == len(props['shape_after_cropping_and_before_resampling']) else \
                    [props['spacing'][0], *_cm.spacing]
                do_sep = do_sep or determine_do_sep_z_and_axis(None, current_spacing, props['spacing'])[0]
            streamed = not do_sep and not any(l.has_regions for l in label_managers[:n])
        except Exception:
            streamed = False
        if not streamed:
            logger.info("[predict_and_fuse] separate-z resampling or region labels: falling back to full-volume export")

    def iter_logits():
        for predictor, dct in zip(predictors, preprocessed_dicts):
            data = dct['data']
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(np.ascontiguousarray(data))
            logits = predictor.predict_logits_from_preprocessed_data(data, reload_model_weight=False, to_cpu=True).float()
            dct['data'] = None  # free preprocessed data as soon as it is consumed
            del data
            yield logits
            # on resume, drop this frame's reference before the next model predicts;
            # otherwise the previous logits stay alive throughout that prediction
            del logits

    if fuse_logits:
        fused = None
        for logits in iter_logits():
            logits = logits.half() if fp16 else logits
            fused = logits if fused is None else fused + logits
            del logits
        fused /= n
        if streamed:
            return _streamed_seg_from_sources(iter([fused]), 1, pm, cm, lm, props, fp16=fp16)
        prob = convert_predicted_logits_to_prob_with_correct_shape(fused.float(), pm, cm, lm, props)
        del fused
        return prob_to_seg(prob)

    if streamed:
        return _streamed_seg_from_sources(iter_logits(), n, pm, cm, lm, props, fp16=fp16)

    prob = None
    for logits, _pm, _cm, _lm, dct in zip(iter_logits(), plans_managers, configuration_managers, label_managers, preprocessed_dicts):
        _prob = convert_predicted_logits_to_prob_with_correct_shape(logits, _pm, _cm, _lm, dct.get('data_properties', {}))
        del logits
        if fp16:
            _prob = _prob.astype(np.float16)
        prob = _prob if prob is None else prob + _prob
        del _prob
    prob /= n
    return prob_to_seg(prob)

def preprocess_worker(queue1, fnames, in_dir, out_dir, suffix, output_ext, plans_managers, dataset_jsons, configuration_managers, tmp_folder, verbose=False):
    """
    Preprocess images and save per-sample preprocessed dicts into compressed npz files in tmp_folder.
    Put only the npz path and lightweight metadata into queue1.
    """
    assert len(plans_managers) == len(configuration_managers) == len(dataset_jsons)
    os.makedirs(tmp_folder, exist_ok=True)
    for fname in fnames:
        try:
            input_file = os.path.join(in_dir, fname + suffix)
            output_file = os.path.join(out_dir, fname + output_ext)
            if verbose:
                print(f"[preprocess_worker] Preprocessing {input_file} ...")

            # fname may contain subfolder, we only want the base name here
            casename = os.path.basename(fname).replace(suffix, '')

            image = sitk.ReadImage(input_file)
            input_array = sitk.GetArrayFromImage(image).astype(np.float32)
            input_shape = input_array.shape
            if verbose:
                print(f"[preprocess_worker] ITK image shape: {input_shape}")
            input_array = input_array[None]  # insert batch dimension
            spacing_for_nnunet = list(image.GetSpacing())[::-1]
            image_properties = { 'spacing': spacing_for_nnunet }

            preprocessed_storage = []  # will hold (data, data_properties) tuples
            for pm, dj, cm in zip(plans_managers, dataset_jsons, configuration_managers):
                ppa = PreprocessAdapterFromNpy([input_array], [None], [image_properties], [None], pm, dj, cm, num_threads_in_multithreaded=1, verbose=verbose)
                dct = next(ppa)
                # ensure data is numpy array on CPU and contiguous
                data = np.ascontiguousarray(dct['data'])
                data_props = dct.get('data_properties', {})
                preprocessed_storage.append((data, data_props))

            # Save to compressed npz.
            # Hash the full (possibly nested) fname so files with the same
            # basename in different subfolders don't collide in tmp_folder.
            fname_hash = hashlib.md5(fname.encode()).hexdigest()[:8]
            tmp_path = os.path.join(tmp_folder, f"{casename}_{fname_hash}_pp.npz")
            save_dict = {}
            for idx, (data, props) in enumerate(preprocessed_storage):
                save_dict[f"p{idx}_data"] = data
                save_dict[f"p{idx}_props"] = json.dumps(props)
            np.savez_compressed(tmp_path, **save_dict)

            # Put only light info into queue
            queue1.put({
                'preprocessed_npz': tmp_path,
                'casename': casename,
                'input_file': input_file,
                'output_file': output_file,
                'out_dir': os.path.join(out_dir, casename)
            })
            del preprocessed_storage, save_dict
        except Exception as e:
            logger.info(f"[preprocess_worker] Error processing {input_file}: {e}")
            traceback.print_exc()

def limit_gpu_memory(target_gb, device_index=0):
    """Cap the CUDA caching allocator of THIS process to ~target_gb gigabytes.

    torch.cuda.set_per_process_memory_fraction is per-process, and calling it
    initializes CUDA — which must NOT happen in the parent before forking GPU
    workers (forked children cannot re-initialize CUDA). Therefore call this
    inside each process that runs inference: the inference workers in parallel
    mode, or the main process in sequential mode.
    """
    total_mem = torch.cuda.get_device_properties(device_index).total_memory
    fraction = min(max((target_gb * 1024**3) / total_mem, 0.0), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    logger.info(f"[GPU {device_index}] Limiting memory to ~{target_gb:.2f} GB "
                f"({fraction * 100:.1f}% of {total_mem / 1024**3:.2f} GB total).")


@torch.inference_mode()
def inference_worker(model_cfg, predictor_class, use_mirroring, queue1, queue2, tmp_folder, verbose=False, device_id=0, gpu_limit_GB=None, fuse_logits=False, fp16=False):
    """
    Created in child process. Build predictors here (do not receive them from parent).
    Load preprocessed npz from tmp_folder, run predictors, push seg to queue2.

    fuse_logits=True averages the models' logits on the preprocessed grid and runs
    resample+softmax+crop-revert once, instead of once per model. Requires all models
    to share the same preprocessing geometry (target spacing, transpose, cropping).
    Note the fusion semantics change from arithmetic mean of probabilities to
    softmax of mean logits.
    """
    try:
        if gpu_limit_GB:
            limit_gpu_memory(gpu_limit_GB, device_id)
        # Create predictors inside this process
        predictors = get_predictors(model_cfg=model_cfg, predictor_class=predictor_class, use_mirroring=use_mirroring, device_id=device_id)
    except Exception as e:
        logger.info(f"[inference_worker] Error creating predictors: {e}")
        traceback.print_exc()
        predictors = []

    # Build managers from predictors locally
    plans_managers = [p.plans_manager for p in predictors]
    configuration_managers = [p.configuration_manager for p in predictors]
    label_managers = [p.label_manager for p in predictors]
    try:
        while True:
            item = queue1.get()
            if item is None:
                break  # Exit signal
            npz_path = item['preprocessed_npz']
            casename = item['casename']
            input_file = item['input_file']
            output_file = item['output_file']
            out_dir = item['out_dir']

            if verbose:
                print(f"[inference_worker] Loading preprocessed npz {npz_path} ...")

            preprocessed_dicts = []
            try:
                with np.load(npz_path, allow_pickle=True) as npz:
                    preprocessed_dicts = []
                    idx = 0
                    while True:
                        key_data = f"p{idx}_data"
                        key_props = f"p{idx}_props"
                        if key_data not in npz:
                            break

                        # Load numpy array
                        np_data = npz[key_data]
                        # Ensure C-contiguous and float32 (predictors usually expect float)
                        np_data = np.ascontiguousarray(np_data).astype(np.float32)

                        # Convert to torch.Tensor on CPU (no CUDA here)
                        try:
                            tensor_data = torch.from_numpy(np_data)
                        except Exception:
                            # Fallback: create tensor via torch.tensor (slower but robust)
                            tensor_data = torch.tensor(np_data, dtype=torch.float32)

                        # If your predictor expects batch dim or channel ordering, keep it as saved.
                        # (Do not .to('cuda') here; the predictor handles device placement / to_cpu flag.)
                        props_json = npz[key_props].tolist()
                        try:
                            data_props = json.loads(props_json)
                        except Exception:
                            data_props = {}

                        preprocessed_dicts.append({'data': tensor_data, 'data_properties': data_props})
                        idx += 1

            except Exception as e:
                logger.info(f"[inference_worker] Failed to load npz {npz_path}: {e}")
                traceback.print_exc()
                continue

            # Optionally delete tmp npz to free space
            try:
                os.remove(npz_path)
            except Exception:
                pass

            if verbose:
                print(f"[inference_worker] Predicting {input_file} ...")
                print(f"\tcasename: {casename}")

            seg = predict_and_fuse(predictors, preprocessed_dicts, plans_managers, configuration_managers, label_managers, fuse_logits=fuse_logits, fp16=fp16, verbose=verbose)
            if seg is None:
                logger.info("[inference_worker] No predictors or no preprocessed dicts, skipping sample")
                continue

            queue2.put( {'seg': seg, 'casename': casename, 'input_file': input_file, 'output_file': output_file, 'out_dir': out_dir} )
            del seg
    except Exception as e:
        logger.info(f"[inference_worker] Error during inference: {e}")
        traceback.print_exc()

@torch.inference_mode()
def post_inference_worker(queue2, global_cnt, total_cnt, start_time, post_process=True, prune_fn_name='rm_small', post_process_func=None, verbose=False):
    try:
        while True:
            item = queue2.get()
            if item is None:
                break  # Exit signal
            seg = item['seg']
            casename = item['casename']
            input_file = item['input_file']
            output_file = item['output_file']
            out_dir = item['out_dir']

            if verbose:
                print(f"[post_inference_worker] Post-processing {input_file} ...")
                print(f"\tcasename: {casename}")

            if verbose:
                print(f"[post_inference_worker] Ensemble fused seg shape: {seg.shape}")

            if post_process:
                if verbose:
                    print(f"[post_inference_worker] Post Process Segmentation")
                if post_process_func is None:
                    post_process_func = generic_prune_nparray
                elif isinstance(post_process_func, str):
                    post_process_func = globals()[post_process_func]
                seg = post_process_func(seg)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            result_image = sitk.GetImageFromArray(seg)
            image = sitk.ReadImage(input_file)
            result_image.CopyInformation(image)
            compressor = "LZW" if output_file.endswith('.tif') or output_file.endswith('.tiff') else ""
            sitk.WriteImage(result_image, output_file, useCompression=True, compressor=compressor)

            # Increment counter (Manager.Value doesn't need get_lock())
            current_count = global_cnt.value + 1
            global_cnt.value = current_count

            eta = (time.time() - start_time) / current_count * (total_cnt - current_count)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(f"[post_inference_worker] Saved to {output_file}, {current_count}/{total_cnt} done, {time.time()-start_time :.0f}s, ETA: {eta_str}")

    except Exception as e:
        logger.info(f"[post_inference_worker] Error during post-processing: {e}")
        traceback.print_exc()

def generic_prune_labels(
    segmentation: np.ndarray,
    labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
    background_label: int = 0,
    filter_fn = None
) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)

    mask_keep = generic_filter_components(mask, filter_fn)

    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret

def keep_topk_components_rm_small(component_ids, component_sizes, k=1, min_size=10):
    assert k >= 1, f"k should be integer >=1, got {k}"
    ids_sizes = sorted(zip(component_ids, component_sizes), key=lambda x: x[1], reverse=True)
    return [ids_sizes[i][0] for i in range(min(k, len(component_ids))) if ids_sizes[i][1] > min_size]

def rm_small(component_ids, component_sizes, min_size=10):
    return [_id for _id, size in zip(component_ids, component_sizes) if size > min_size]

def generic_prune_nparray(seg, prune_fn_name='rm_small'):
    """Actually prune_fn_name can also be a function directly."""
    filter_fn = globals()[prune_fn_name] if isinstance(prune_fn_name, str) else prune_fn_name
    labels = np.unique(seg)
    for label in labels:
        if label == 0:
            continue
        seg = generic_prune_labels(seg, label, filter_fn=filter_fn)
    return seg

def _erode_dilate_seg(seg_array, iterations=1):
    processed_seg = np.zeros_like(seg_array)
    for label in np.unique(seg_array):
        if label == 0:
            continue
        binary_mask = (seg_array == label).astype(np.uint8)
        # Erode
        eroded_mask = scipy.ndimage.binary_erosion(binary_mask, iterations=iterations)
        # Dilate
        dilated_mask = scipy.ndimage.binary_dilation(eroded_mask, iterations=iterations)
        processed_seg[dilated_mask > 0] = label
    return processed_seg

def _erode_seg(seg_array, iterations=1):
    processed_seg = np.zeros_like(seg_array)
    for label in np.unique(seg_array):
        if label == 0:
            continue
        binary_mask = (seg_array == label).astype(np.uint8)
        # Erode
        eroded_mask = scipy.ndimage.binary_erosion(binary_mask, iterations=iterations)
        processed_seg[eroded_mask > 0] = label
    return processed_seg

def _dilate_seg(seg_array, iterations=1):
    processed_seg = np.zeros_like(seg_array)
    for label in np.unique(seg_array):
        if label == 0:
            continue
        binary_mask = (seg_array == label).astype(np.uint8)
        # Dilate
        dilated_mask = scipy.ndimage.binary_dilation(binary_mask, iterations=iterations)
        processed_seg[dilated_mask > 0] = label
    return processed_seg

def _erode_dilate_rm_small(seg, iterations=1, min_size=10):
    seg = _erode_dilate_seg(seg, iterations=iterations)
    seg = generic_prune_nparray(seg, prune_fn_name=partial(rm_small, min_size=min_size))
    return seg

def _erode_rm_small(seg, iterations=1, min_size=10):
    seg = _erode_seg(seg, iterations=iterations)
    seg = generic_prune_nparray(seg, prune_fn_name=partial(rm_small, min_size=min_size))
    return seg

def _dilate_rm_small(seg, iterations=1, min_size=10):
    seg = _dilate_seg(seg, iterations=iterations)
    seg = generic_prune_nparray(seg, prune_fn_name=partial(rm_small, min_size=min_size))
    return seg

def _dilate_erode_rm_small(seg, iterations=1, min_size=10):
    seg = _dilate_seg(seg, iterations=iterations)
    seg = _erode_seg(seg, iterations=iterations)
    seg = generic_prune_nparray(seg, prune_fn_name=partial(rm_small, min_size=min_size))
    return seg

def _rm_small_dilate_erode(seg, iterations=1, min_size=10):
    seg = generic_prune_nparray(seg, prune_fn_name=partial(rm_small, min_size=min_size))
    seg = _dilate_seg(seg, iterations=iterations)
    seg = _erode_seg(seg, iterations=iterations)
    return seg

open1 = _erode_dilate_seg
open1_rm1000 = partial(_erode_dilate_rm_small, iterations=1, min_size=1000)
open1_rm5000 = partial(_erode_dilate_rm_small, iterations=1, min_size=5000)
open1_rm10000 = partial(_erode_dilate_rm_small, iterations=1, min_size=10000)
open1_rm50000 = partial(_erode_dilate_rm_small, iterations=1, min_size=50000)
erode1_rm50000 = partial(_erode_rm_small, iterations=1, min_size=50000)
erode2_rm50000 = partial(_erode_rm_small, iterations=2, min_size=50000)

dilate1_rm5000 = partial(_dilate_rm_small, iterations=1, min_size=5000)
close1_rm5000 = partial(_dilate_erode_rm_small, iterations=1, min_size=5000)
close1_rm50000 = partial(_dilate_erode_rm_small, iterations=1, min_size=50000)
rm5000_close1 = partial(_rm_small_dilate_erode, iterations=1, min_size=5000)
rm10000_close1 = partial(_rm_small_dilate_erode, iterations=1, min_size=10000)
rm20000_close1 = partial(_rm_small_dilate_erode, iterations=1, min_size=20000)
rm50000_close1 = partial(_rm_small_dilate_erode, iterations=1, min_size=50000)
rm1000 = partial(generic_prune_nparray, prune_fn_name=partial(rm_small, min_size=1000))
rm5000 = partial(generic_prune_nparray, prune_fn_name=partial(rm_small, min_size=5000))

def post_process_file(input_file, output_file, post_process_func=None, verbose=False):
    """
    Post-process the segmentation nifti/tiff file.
    Remove small components and keep the largest component for each label.
    """
    if verbose:
        print(f"Post-processing {input_file}...")
    seg = sitk.ReadImage(input_file)
    seg_array = sitk.GetArrayFromImage(seg).astype(np.uint8)

    # post-process
    if post_process_func is None:
        post_process_func = generic_prune_nparray
    elif isinstance(post_process_func, str):
        post_process_func = globals()[post_process_func]
    seg_array = post_process_func(seg_array)

    # save
    result_image = sitk.GetImageFromArray(seg_array)
    result_image.CopyInformation(seg)
    compressor = "LZW" if output_file.endswith('.tif') or output_file.endswith('.tiff') else ""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sitk.WriteImage(result_image, output_file, useCompression=True, compressor=compressor)
    if verbose:
        print(f"Post-processed segmentation saved to {output_file}")

def post_process_folder(in_dir, out_dir, suffix='.nii.gz', post_process_func=None):
    """
    Post-process all segmentation nifti/tiff files in the input directory.
    Remove small components and keep the largest component for each label.
    """
    st = time.time()
    logger.info(f"Running post-processing on {in_dir}...")
    fnames = subfiles(folder=in_dir, join=False, suffix=suffix)
    for i, fname in enumerate(tqdm(fnames, desc="Post-processing")):
        input_file = os.path.join(in_dir, fname)
        output_file = os.path.join(out_dir, fname)
        post_process_file(input_file, output_file, post_process_func=post_process_func)
    logger.info(f"Post-processing done, {time.time()-st :.0f}s")

@torch.inference_mode()
def infer_one_sample(input_file, output_file, casename, predictors, post_process=True, prune_fn_name='rm_small', post_process_func=None, fuse_logits=False, fp16=False):
    print(f"Predicting {input_file} ...")
    print(f"\tcasename: {casename}")

    image = sitk.ReadImage(input_file)
    input_array = sitk.GetArrayFromImage(image).astype(np.float32)

    input_shape = input_array.shape
    print(f"[*] ITK image shape: {input_shape}")
    input_array = input_array[None]  # insert batch dimension
    spacing_for_nnunet=list(image.GetSpacing())[::-1]
    props = { 'spacing': spacing_for_nnunet }
    print(f"[*] ITK image spacing after x-z Transposed: {spacing_for_nnunet}")

    print("[*] preprocessing...")
    preprocessed_dicts = []
    for predictor in predictors:
        ppa = PreprocessAdapterFromNpy([input_array], [None], [props], [None],
                                       predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=False)
        preprocessed_dicts.append(next(ppa))

    print("[*] prediction...")
    del input_array  # the preprocessed copies are what matters from here on
    seg = predict_and_fuse(
        predictors, preprocessed_dicts,
        [p.plans_manager for p in predictors],
        [p.configuration_manager for p in predictors],
        [p.label_manager for p in predictors],
        fuse_logits=fuse_logits, fp16=fp16)
    print(f"[*] Ensemble fused seg shape: {seg.shape}")

    if post_process:
        print(f"[*] Post Process Segmentation")
        if post_process_func is None:
            post_process_func = generic_prune_nparray
        elif isinstance(post_process_func, str):
            post_process_func = globals()[post_process_func]
        seg = post_process_func(seg)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_image = sitk.GetImageFromArray(seg)
    result_image.CopyInformation(image)
    compressor = "LZW" if output_file.endswith('.tif') or output_file.endswith('.tiff') else ""
    sitk.WriteImage(result_image, output_file, useCompression=True, compressor=compressor)
    print(f"[*] Saved to {output_file}")

def infer_folder(
    in_dir, out_dir, fnames=None, split_file=None, split_fold=None,
    suffix='_0000.nii.gz', output_ext='.nii.gz',
    n_rand_samples=None,
    model_cfg=None, predictor_class=nnUNetPredictor, use_mirroring=True,
    post_process=True,
    prune_fn_name='rm_small',
    post_process_func=None,
    skip_existing=False,
    sequential=False,
    queue1_size=12, queue2_size=12, n_preprocess_workers=6, n_infer_workers=1, n_post_inference_workers=6,
    n_gpus=1, gpu_limit_GB=None, fuse_logits=False, fp16=False
):
    st0 = time.time()

    # Only create predictors here if running sequentially (so we avoid pickling GPU objects).
    if sequential:
        # Sequential: inference runs in this process, so the cap is set here. In parallel
        # mode it is set inside each inference_worker instead (setting it here would
        # initialize CUDA in the parent and break the forked GPU workers).
        if gpu_limit_GB:
            limit_gpu_memory(gpu_limit_GB, 0)
        predictors = get_predictors(model_cfg=model_cfg, predictor_class=predictor_class, use_mirroring=use_mirroring)
    else:
        predictors = None

    if fnames is not None:
        logger.info(f"NO. file names: {len(fnames)}")
    elif split_file:
        fnames = load_json(split_file)[split_fold]['val']
        logger.info(f"Found {len(fnames)} val files in fold_{split_fold} of split_file {split_file}")
    else:
        fnames = subfiles(folder=in_dir, join=False, suffix=suffix)
        fnames = [name.replace(suffix, '') for name in fnames]
        logger.info(f"Found {len(fnames)} matching files in {in_dir}")

    if n_rand_samples is not None and n_rand_samples > 0:
        n_total = len(fnames)
        fnames = sorted(fnames)  # Sort to ensure reproducibility
        random.seed(42)  # For reproducibility
        random.shuffle(fnames)
        fnames = sorted(fnames[:n_rand_samples])
        logger.info(f"Randomly selected {n_rand_samples} samples from {n_total} files")
        print('\t' + '\n\t'.join(fnames) + '\n')

    if skip_existing:
        n_before = len(fnames)
        fnames = [name for name in fnames if not isfile(os.path.join(out_dir, name + output_ext))]
        n_skipped = n_before - len(fnames)
        logger.info(f"skip_existing=True: skipped {n_skipped} files with existing prediction, {len(fnames)} remaining")
        if len(fnames) == 0:
            logger.info("All predictions already exist, nothing to do.")
            return

    if sequential:
        for i, fname in enumerate(fnames):
            st = time.time()
            input_file = os.path.join(in_dir, fname+suffix)
            output_file = os.path.join(out_dir, fname+output_ext)
            casename = os.path.basename(input_file).replace(suffix, '')
            infer_one_sample(input_file, output_file, casename, predictors, post_process=post_process, prune_fn_name=prune_fn_name, post_process_func=post_process_func, fuse_logits=fuse_logits, fp16=fp16)
            logger.info(f"Infer {i+1}/{len(fnames)} done, {time.time()-st :.0f}s")
    else:
        # Use multiprocessing with queues for parallel processing
        from multiprocessing import Process, Queue, Manager

        queue1 = Queue(maxsize=queue1_size)
        queue2 = Queue(maxsize=queue2_size)
        manager = Manager()
        global_cnt = manager.Value('i', 0)

        # keep managers (lightweight) to allow ppa to run consistently
        # note: these are simple python objects (dictionaries/structs) from predictors; okay to pass
        # but do NOT pass full predictor objects
        # We still extract these from a temporary predictors instance to ensure they exist
        sample_predictors = get_predictors(model_cfg=model_cfg, predictor_class=predictor_class, use_mirroring=use_mirroring)
        plans_managers = [predictor.plans_manager for predictor in sample_predictors]
        dataset_jsons = [predictor.dataset_json for predictor in sample_predictors]
        configuration_managers = [predictor.configuration_manager for predictor in sample_predictors]
        label_managers = [predictor.label_manager for predictor in sample_predictors]
        # We will NOT pass sample_predictors to children; free it
        del sample_predictors

        start_time = time.time()

        # Distribute files among preprocessing workers
        fnames_per_worker = []
        n_pre = max(1, n_preprocess_workers)
        chunk_size = max(1, len(fnames) // n_pre)
        for i in range(n_pre):
            if i == n_pre - 1:
                fnames_per_worker.append(fnames[i * chunk_size:])
            else:
                fnames_per_worker.append(fnames[i * chunk_size:(i + 1) * chunk_size])

        # tmp folder for preprocessed npz
        tmp_folder = os.path.join(out_dir, "tmp_preprocessed")
        os.makedirs(tmp_folder, exist_ok=True)

        # Create and start preprocessing workers
        preprocess_processes = []
        for worker_fnames in fnames_per_worker:
            if len(worker_fnames) > 0:
                p = Process(target=preprocess_worker, args=(queue1, worker_fnames, in_dir, out_dir, suffix, output_ext, plans_managers, dataset_jsons, configuration_managers, tmp_folder))
                p.start()
                preprocess_processes.append(p)

        # Create and start inference workers (create predictors inside each process)
        infer_processes = []
        for i in range(max(1, n_infer_workers)):
            # Distribute inference workers onto GPUs in round-robin fashion
            p = Process(target=inference_worker, args=(model_cfg, predictor_class, use_mirroring, queue1, queue2, tmp_folder, False, i%n_gpus, gpu_limit_GB, fuse_logits, fp16))
            p.start()
            infer_processes.append(p)

        # Create and start a pool of post_inference workers
        post_processes = []
        for _ in range(max(1, n_post_inference_workers)):
            p = Process(target=post_inference_worker, args=(queue2, global_cnt, len(fnames), start_time, post_process, prune_fn_name, post_process_func))
            p.start()
            post_processes.append(p)

        # Wait for all preprocessing processes to complete
        for p in preprocess_processes:
            p.join()

        # Add exit signals for inference workers
        for _ in range(len(infer_processes)):
            queue1.put(None)

        # Wait for inference workers to finish
        for p in infer_processes:
            p.join()

        # Add exit signals for post_inference workers
        for _ in range(len(post_processes)):
            queue2.put(None)

        # Wait for post_inference workers to finish
        for p in post_processes:
            p.join()

        # Cleanup tmp folder (optional): remove remaining npz files
        try:
            for f in os.listdir(tmp_folder):
                fp = os.path.join(tmp_folder, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            # optionally rmdir
            try:
                os.rmdir(tmp_folder)
            except Exception:
                pass
        except Exception:
            pass

    logger.info(f"Infer done, {time.time()-st0 :.0f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequential', type=parse_bool, default=False, help='Run sequentially without multiprocessing')
    parser.add_argument('--n_preprocess_workers', type=int, default=6, help='Number of preprocessing workers')
    parser.add_argument('--n_infer_workers', type=int, default=1, help='Number of inference workers (should be 1 to avoid CUDA issues)')
    parser.add_argument('--n_post_inference_workers', type=int, default=6, help='Number of post_inference workers')
    parser.add_argument('--queue1_size', type=int, default=12, help='Size of queue 1 between preprocessing and inference')
    parser.add_argument('--queue2_size', type=int, default=12, help='Size of queue 2 between inference and post_inference')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs available for inference, >= 1')

    parser.add_argument('--num_fg_classes', type=int, required=True, help='Number of foreground classes')
    parser.add_argument('--base_model_dir', type=str, required=True, help='Base model directory containing trained model subdirectories')
    parser.add_argument('--model_subdir', type=str, required=True, help='Model subdirectory within the base model directory')
    parser.add_argument('--model_fold', type=int, required=True, help='Fold number of model')
    parser.add_argument('--ckpt_name', type=str, default='checkpoint_final.pth', help='Checkpoint name of model')
    parser.add_argument('--tile_step_size', type=float, default=0.5, help='Tile step size for inference')
    parser.add_argument('--use_mirroring', type=parse_bool, default=True, help='Whether to enable mirroring if the checkpoint configuration allows it')
    parser.add_argument('--post_process', type=parse_bool, default=False, help='Whether to post process segmentation, e.g. remove small components')
    parser.add_argument('--prune_fn_name', type=str, required=False, default='rm_small', help='Prune function name')
    parser.add_argument('--post_process_func', type=str, required=False, default=None, help='Post process function name')
    parser.add_argument('--skip_existing', type=parse_bool, default=False, help='Skip predicting for samples whose output prediction already exists')

    parser.add_argument('--in_dir', type=str, required=True, help='Input directory containing test images')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save segmentations')
    parser.add_argument('--split_file', type=str, required=False, default=None, help='[Optional] Split file to specify input images')
    parser.add_argument('--split_fold', type=int, required=False, default=0, help='[Optional] Fold number in the split file, where we get the val set to predict on')
    parser.add_argument('--suffix', type=str, required=False, default='_0000.nii.gz', help='Suffix for input images')
    parser.add_argument('--output_ext', type=str, required=False, default='.nii.gz', help='Extension for output prediction')

    args = parser.parse_args()

    model_cfg = [
        dict(base_model_dir=args.base_model_dir, subdir=args.model_subdir, model_fold=args.model_fold, ckpt_name=args.ckpt_name, patch_size=None, tile_step_size=args.tile_step_size),
    ]

    infer_folder(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        split_file=args.split_file,
        split_fold=args.split_fold,
        suffix=args.suffix,
        output_ext=args.output_ext,
        model_cfg=model_cfg,
        sequential=args.sequential,
        n_preprocess_workers=args.n_preprocess_workers,
        n_infer_workers=args.n_infer_workers,
        n_post_inference_workers=args.n_post_inference_workers,
        queue1_size=args.queue1_size,
        queue2_size=args.queue2_size,
        use_mirroring=args.use_mirroring,
        post_process=args.post_process,
        prune_fn_name=args.prune_fn_name,
        post_process_func=args.post_process_func,
        skip_existing=args.skip_existing,
        n_gpus=args.n_gpus
    )
