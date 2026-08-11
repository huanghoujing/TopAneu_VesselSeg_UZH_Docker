from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def _convert_to_npy(b2nd_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    import blosc2
    data_npy = b2nd_file[:-5] + ".npy"
    seg_b2nd = b2nd_file[:-5] + "_seg.b2nd"
    seg_npy = b2nd_file[:-5] + "_seg.npy"
    try:
        if overwrite_existing or not isfile(data_npy):
            try:
                data = blosc2.open(urlpath=b2nd_file, mode='r')[:]
            except Exception as e:
                print(f"Unable to open preprocessed file {b2nd_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, data)

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                seg = blosc2.open(urlpath=seg_b2nd, mode='r')[:]
            except Exception as e:
                print(f"Unable to open preprocessed file {seg_b2nd}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(seg_npy, seg)

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(b2nd_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify_npy: bool = False):
    """
    all .b2nd files in this folder belong to the dataset, unpack them all (data + seg) to .npy
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        b2nd_files = subfiles(folder, True, None, ".b2nd", True)
        # exclude the per-case segmentation files; _convert_to_npy derives the _seg.b2nd path itself
        b2nd_files = [f for f in b2nd_files if not f.endswith("_seg.b2nd")]
        p.starmap(_convert_to_npy, zip(b2nd_files,
                                       [unpack_segmentation] * len(b2nd_files),
                                       [overwrite_existing] * len(b2nd_files),
                                       [verify_npy] * len(b2nd_files))
                  )


def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all .b2nd data files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-5] for i in os.listdir(folder)
                        if i.endswith(".b2nd") and not i.endswith("_seg.b2nd") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')