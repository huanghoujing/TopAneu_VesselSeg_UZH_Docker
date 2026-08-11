import math
import os
from typing import List, Optional, Sequence, Tuple

import blosc2
import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle
from nnunetv2.training.dataloading.utils import get_case_identifiers


def comp_blosc2_params(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    bytes_per_pixel: int = 4,
    max_block_nbytes: int = 128 * 1024,
    max_chunk_nbytes: int = 6 * 1024**2,
    max_chunk_to_patch_ratio_per_axis: Optional[float] = 1.5,
    grow_singleton_patch_axes: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Compute Blosc2 block and chunk shapes for 4D arrays with layout:

        image_size = (c, x, y, z)

    patch_size is spatial only:

        patch_size = (x, y, z) for 3D patches
        patch_size = (y, z)    for 2D patches, internally promoted to (1, y, z)

    The channel axis is always kept at 1 for blocks and chunks.

    Ported verbatim from official nnUNet (nnunetv2/training/dataloading/nnunet_dataset.py).
    """

    def ceil_pow2(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def prev_pow2_below(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << ((n - 1).bit_length() - 1)

    def nbytes(shape: Sequence[int]) -> int:
        return math.prod(int(i) for i in shape) * int(bytes_per_pixel)

    if len(image_size) != 4:
        raise ValueError(f"image_size must be 4D, i.e. (c, x, y, z). Got {image_size}.")

    if len(patch_size) not in (2, 3):
        raise ValueError(f"patch_size must be 2D or 3D spatial shape. Got {patch_size}.")

    if bytes_per_pixel <= 0:
        raise ValueError("bytes_per_pixel must be positive.")

    if max_block_nbytes <= 0:
        raise ValueError("max_block_nbytes must be positive.")

    if max_chunk_nbytes <= 0:
        raise ValueError("max_chunk_nbytes must be positive.")

    if (
        max_chunk_to_patch_ratio_per_axis is not None
        and max_chunk_to_patch_ratio_per_axis < 1.0
    ):
        raise ValueError("max_chunk_to_patch_ratio_per_axis must be >= 1.0 or None.")

    image_size = tuple(int(i) for i in image_size)

    if any(i <= 0 for i in image_size):
        raise ValueError(f"All image_size entries must be positive. Got {image_size}.")

    is_2d_patch = len(patch_size) == 2

    if is_2d_patch:
        patch_spatial = (1, int(patch_size[0]), int(patch_size[1]))
    else:
        patch_spatial = tuple(int(i) for i in patch_size)

    if any(i <= 0 for i in patch_spatial):
        raise ValueError(f"All patch_size entries must be positive. Got {patch_size}.")

    image_spatial = image_size[1:]

    if max_chunk_to_patch_ratio_per_axis is None:
        target_spatial = image_spatial
    else:
        target_spatial = tuple(
            min(
                image_spatial[ax],
                int(math.floor(max_chunk_to_patch_ratio_per_axis * patch_spatial[ax])),
            )
            for ax in range(3)
        )

    target_spatial = tuple(max(1, int(i)) for i in target_spatial)

    # --------------------
    # Block shape
    # --------------------
    block = [1]
    for p, img in zip(patch_spatial, image_spatial):
        block.append(min(ceil_pow2(p), img))

    effective_block_cap = min(max_block_nbytes, max_chunk_nbytes)

    while nbytes(block) > effective_block_cap:
        candidate_axes = [ax for ax in range(3) if block[ax + 1] > 1]

        if not candidate_axes:
            raise ValueError(
                f"Cannot fit minimum block {tuple(block)} into "
                f"{effective_block_cap} bytes with bytes_per_pixel={bytes_per_pixel}."
            )

        picked_axis = max(
            candidate_axes,
            key=lambda ax: (block[ax + 1] / target_spatial[ax], ax),
        )

        block[picked_axis + 1] = max(1, prev_pow2_below(block[picked_axis + 1]))

    # --------------------
    # Chunk shape
    # --------------------
    chunk = block.copy()

    while True:
        candidates = []

        for ax in range(3):
            if is_2d_patch and ax == 0:
                continue

            if patch_spatial[ax] == 1 and not grow_singleton_patch_axes:
                continue

            if chunk[ax + 1] >= image_spatial[ax]:
                continue

            if chunk[ax + 1] >= target_spatial[ax]:
                continue

            candidate = chunk.copy()

            candidate[ax + 1] = min(
                candidate[ax + 1] + block[ax + 1],
                image_spatial[ax],
                target_spatial[ax],
            )

            if candidate[ax + 1] == chunk[ax + 1]:
                continue

            candidate_nbytes = nbytes(candidate)

            if candidate_nbytes > max_chunk_nbytes:
                continue

            coverage_ratio = chunk[ax + 1] / target_spatial[ax]

            candidates.append(
                (
                    coverage_ratio,
                    -image_spatial[ax],
                    ax,
                    -candidate_nbytes,
                    candidate,
                )
            )

        if not candidates:
            break

        _, _, _, _, chunk = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )

    return tuple(int(i) for i in block), tuple(int(i) for i in chunk)


class nnUNetDataset(object):
    """blosc2-backed dataset (hard switch from .npz).

    Storage layout per case in the preprocessed folder:
        {c}.b2nd        -> data    (blosc2)
        {c}_seg.b2nd    -> seg     (blosc2)
        {c}.pkl         -> properties (pickle, unchanged)

    Like before, we keep an optional unpacked-to-.npy fast path: `unpack_dataset` writes {c}.npy and
    {c}_seg.npy (memory-mapped during training), and `load_case` prefers them when present. This keeps
    the exact `(data, seg, properties)` 3-tuple contract and numpy return type so all existing
    dataloaders and `perform_actual_validation` (torch.from_numpy(data)) work unchanged.
    """
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__()
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.b2nd")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.b2nd")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))

        blosc2.set_nthreads(1)
        # mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
        self.mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def _open_b2nd(self, urlpath):
        return blosc2.open(urlpath=urlpath, mode='r', dparams={'nthreads': 1}, **self.mmap_kwargs)[:]

    def load_case(self, key):
        entry = self[key]
        data_b2nd = entry['data_file']            # {c}.b2nd
        data_npy = data_b2nd[:-5] + ".npy"
        seg_b2nd = data_b2nd[:-5] + "_seg.b2nd"
        seg_npy = data_b2nd[:-5] + "_seg.npy"

        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
        elif isfile(data_npy):
            data = np.load(data_npy, 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = self._open_b2nd(data_b2nd)

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
        elif isfile(seg_npy):
            seg = np.load(seg_npy, 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
        else:
            seg = self._open_b2nd(seg_b2nd)

        if 'seg_from_prev_stage_file' in entry.keys():
            prev_b2nd = entry['seg_from_prev_stage_file']
            prev_npy = prev_b2nd[:-5] + ".npy"
            if isfile(prev_npy):
                seg_prev = np.load(prev_npy, 'r')
            else:
                seg_prev = self._open_b2nd(prev_b2nd)
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']

    @staticmethod
    def _select_filter(arr: np.ndarray, blocks, chunks, codec, clevel) -> "blosc2.Filter":
        """Pick the better blosc2 filter (NOFILTER vs SHUFFLE) by trial-compressing a centered slab.
        Ported from official nnUNetDatasetBlosc2._select_filter."""
        try:
            shape = tuple(int(s) for s in arr.shape)
            slab_shape = [min(int(c), s) for c, s in zip(chunks, shape)]
            slices = tuple(slice((s - ss) // 2, (s - ss) // 2 + ss) for s, ss in zip(shape, slab_shape))
            slab = np.ascontiguousarray(arr[slices])
            trial_blocks = tuple(max(1, min(int(b), ss)) for b, ss in zip(blocks, slab_shape))

            best_filter, best_bytes = blosc2.Filter.NOFILTER, None
            for f in (blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE):
                cparams = {'codec': codec, 'clevel': clevel, 'nthreads': 4, 'filters': [f]}
                comp = blosc2.asarray(slab, chunks=tuple(slab_shape), blocks=trial_blocks, cparams=cparams)
                cb = comp.schunk.cbytes
                if best_bytes is None or cb < best_bytes:
                    best_bytes, best_filter = cb, f
            return best_filter
        except Exception as e:
            from warnings import warn
            warn(f'_select_filter failed ({e!r}); falling back to NOFILTER.')
            return blosc2.Filter.NOFILTER

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 5,
            codec=blosc2.Codec.LZ4HC,
            filters=None,
            filters_seg=None,
    ):
        if chunks is None or blocks is None:
            from warnings import warn
            blocks, chunks = comp_blosc2_params(data.shape, (128, 128, 128))
            warn(f'Warning: Received empty chunks or blocks. Computed with comp_blosc2_params. This is bad because we '
                 f'do not know the access pattern here (patch size).\n'
                 f'data shape: {data.shape}\nchunks {chunks}\nblocks {blocks}\n')

        blosc2.set_nthreads(1)

        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        if filters is None:
            data_filters = [nnUNetDataset._select_filter(data, blocks, chunks, codec, clevel)]
        else:
            data_filters = list(filters)

        if filters_seg is None:
            seg_filters = [nnUNetDataset._select_filter(seg, blocks_seg, chunks_seg, codec, clevel)]
        else:
            seg_filters = list(filters_seg)

        cparams = {'codec': codec, 'filters': data_filters, 'nthreads': 4, 'clevel': clevel}
        cparams_seg = {'codec': codec, 'filters': seg_filters, 'nthreads': 4, 'clevel': clevel}

        blosc2.asarray(
            np.ascontiguousarray(data),
            urlpath=output_filename_truncated + '.b2nd',
            chunks=chunks, blocks=blocks, cparams=cparams,
        )
        blosc2.asarray(
            np.ascontiguousarray(seg),
            urlpath=output_filename_truncated + '_seg.b2nd',
            chunks=chunks_seg, blocks=blocks_seg, cparams=cparams_seg,
        )
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str,
            chunks_seg=None,
            blocks_seg=None,
    ):
        if isfile(output_filename_truncated + '.b2nd'):
            os.remove(output_filename_truncated + '.b2nd')
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + '.b2nd',
                       chunks=chunks_seg, blocks=blocks_seg)
