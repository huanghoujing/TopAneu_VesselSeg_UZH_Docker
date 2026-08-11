import os
import blosc2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from skimage.morphology import skeletonize, dilation
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile


def calculate_tubed_skeleton(seg_all, do_tube):
    bin_seg = seg_all > 0
    seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)

    for b in range(bin_seg.shape[0]):
        if np.sum(bin_seg[b, 0]) == 0:
            continue
        skel = skeletonize(bin_seg[b, 0])
        skel = (skel > 0).astype(np.int16)
        if do_tube:
            skel = dilation(dilation(skel))
        skel *= seg_all[b, 0].astype(np.int16)
        seg_all_skel[b, 0] = skel
    return seg_all_skel

def get_ct_mr_modality_and_vessel_presence_9_13(seg, case_identifier):
    modality = 0 if '_ct_' in case_identifier else 1
    target_labels = [9,13]
    if isinstance(seg, torch.Tensor):
        vessel_presence = [1 if torch.isin(seg, x).any() else 0 for x in target_labels]
    else:
        vessel_presence = [1 if np.isin(seg, [x]).any() else 0 for x in target_labels]
    return [modality] + vessel_presence

def get_ct_mr_modality_and_vessel_presence_15_16_31_32(seg, case_identifier):
    modality = 0 if '_ct_' in case_identifier else 1
    target_labels = [15,16,31,32]
    if isinstance(seg, torch.Tensor):
        vessel_presence = [1 if torch.isin(seg, x).any() else 0 for x in target_labels]
    else:
        vessel_presence = [1 if np.isin(seg, [x]).any() else 0 for x in target_labels]
    return [modality] + vessel_presence

class nnUNetDataLoader3DSkelAndCls(object):
    """Self-contained dataloader that mirrors the trainer-facing API and uses shared memory tensors."""

    def __init__(
        self,
        preprocessed_dataset_folder,
        case_identifiers,
        batch_size,
        patch_size,
        final_patch_size,
        label_manager,
        oversample_foreground_percent=0.0,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling=False,
        transforms=None,
        folder_with_segs_from_previous_stage=None,
        num_images_properties_loading_threshold=0,
        modality_presence_func_name=None,
        num_threads=1,
        device=torch.device("cpu"),
        use_skeleton=True,
        use_fg_dilation=True
    ):
        # Initialize dataset dictionary (replaces nnUNetDataset)
        self.dataset = {}
        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        blosc2.set_nthreads(1)
        # mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
        self.mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(preprocessed_dataset_folder, f"{c}.b2nd")
            self.dataset[c]['properties_file'] = join(preprocessed_dataset_folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.b2nd")

        # Load properties into memory if threshold allows
        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for c in self.dataset.keys():
                self.dataset[c]['properties'] = load_pickle(self.dataset[c]['properties_file'])

        self.batch_size = batch_size
        self.patch_size = np.array(patch_size).astype(int)
        self.final_patch_size = np.array(final_patch_size).astype(int)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.sampling_probabilities = sampling_probabilities
        self.pad_sides = pad_sides
        self.label_manager = label_manager
        self.transforms = transforms

        self.list_of_keys = list(self.dataset.keys())
        self.num_channels = None
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.need_to_pad = (self.patch_size - self.final_patch_size).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides

        self.data_shape, self.seg_shape = self.determine_shapes()
        self.get_do_oversample = (
            self._probabilistic_oversampling if probabilistic_oversampling else self._oversample_last_XX_percent
        )
        print(f"{self.__class__.__name__} {modality_presence_func_name = }")
        self.modality_presence_func = globals()[modality_presence_func_name] if modality_presence_func_name is not None else None
        self.num_threads = num_threads
        self.device = device
        self.use_skeleton = use_skeleton
        self.use_fg_dilation = use_fg_dilation

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def __len__(self):
        return len(self.list_of_keys)

    # ------------------------------------------------------------------
    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def load_case(self, key):
        """Load case data, seg, and properties - replaces nnUNetDataset.load_case"""
        entry = self.dataset[key]
        
        # Load properties if not already in memory
        if 'properties' not in entry.keys():
            properties = load_pickle(entry['properties_file'])
        else:
            properties = entry['properties']
        
        data_b2nd = entry['data_file']            # {c}.b2nd
        # Load data (prefer unpacked .npy memmap, fall back to .b2nd)
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
        elif isfile(data_b2nd[:-5] + ".npy"):
            data = np.load(data_b2nd[:-5] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = blosc2.open(urlpath=data_b2nd, mode='r', dparams={'nthreads': 1}, **self.mmap_kwargs)[:]

        # Load segmentation
        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
        elif isfile(data_b2nd[:-5] + "_seg.npy"):
            seg = np.load(data_b2nd[:-5] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
        else:
            seg = blosc2.open(urlpath=data_b2nd[:-5] + "_seg.b2nd", mode='r', dparams={'nthreads': 1}, **self.mmap_kwargs)[:]

        # Load previous stage segmentation if available
        if 'seg_from_prev_stage_file' in entry.keys():
            prev_b2nd = entry['seg_from_prev_stage_file']
            if isfile(prev_b2nd[:-5] + ".npy"):
                seg_prev = np.load(prev_b2nd[:-5] + ".npy", 'r')
            else:
                seg_prev = blosc2.open(urlpath=prev_b2nd, mode='r', dparams={'nthreads': 1}, **self.mmap_kwargs)[:]
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, properties

    def determine_shapes(self):
        data, seg, _ = self.load_case(self.list_of_keys[0])
        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def get_indices(self):
        replace = len(self.list_of_keys) < self.batch_size
        if self.sampling_probabilities is not None:
            return np.random.choice(self.list_of_keys, self.batch_size, replace=True, p=self.sampling_probabilities)
        return np.random.choice(self.list_of_keys, self.batch_size, replace=replace)

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations, overwrite_class=None, verbose: bool = False):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            class_locations = class_locations if class_locations is not None else {}
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations.get(selected_class, [])) == 0:
                    selected_class = None
            elif force_fg:
                if class_locations is None:
                    raise RuntimeError('class_locations missing while oversampling foreground')
                if overwrite_class is not None and overwrite_class not in class_locations.keys():
                    raise AssertionError('desired class has no class_locations')
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])
                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if (
                        overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)
                    ) else overwrite_class
            else:
                raise RuntimeError('Invalid oversampling configuration')

            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None
            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    # ------------------------------------------------------------------
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        skel_all = None

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self.load_case(i)

            shape = data.shape[1:]
            dim = len(shape)
            class_locations = properties.get('class_locations', None)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            data_slice = tuple([slice(0, data.shape[0])] + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg_slice = tuple([slice(0, seg.shape[0])] + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[data_slice]
            seg = seg[seg_slice]

            padding = [(-min(0, bbox_lbs[idx]), max(bbox_ubs[idx] - shape[idx], 0)) for idx in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            torch_nthreads = torch.get_num_threads()
            with torch.no_grad():
                with threadpool_limits(limits=self.num_threads, user_api=None):
                    torch.set_num_threads(self.num_threads)
                    data_all = torch.from_numpy(data_all).float()  #.to(self.device)
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)  #.to(self.device)
                    images = []
                    segs = []
                    skels = []
                    dilated_fg = []
                    modality_presence = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'].to(self.device))
                        segs.append(tmp['segmentation'].to(self.device))
                        if self.use_skeleton:
                            skels.append(tmp['skel'].to(self.device))
                        if self.use_fg_dilation:
                            dilated_fg.append(tmp['dilated_fg'].to(self.device))
                        # Here the returned modality_presence is a list, not tensor
                        modality_presence.append(self.modality_presence_func(seg=seg_all[b], case_identifier=selected_keys[b]) if self.modality_presence_func is not None else -1)
                    assert not isinstance(segs[0], list), "DeepSupervision is not supported, so segs[0] shouldn't be a list"
                    data_all = torch.stack(images).share_memory_()                    
                    seg_all = torch.stack(segs).share_memory_()
                    if self.use_skeleton:
                        skel_all = torch.stack(skels).share_memory_()
                    if self.use_fg_dilation:
                        dilated_fg = torch.stack(dilated_fg).share_memory_()
                    modality_presence = torch.tensor(modality_presence, dtype=torch.long, device=self.device).share_memory_()
                    del segs, images, skels
            torch.set_num_threads(torch_nthreads)

        results = {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'modality_presence': modality_presence}
        if self.use_skeleton:
            results['skel'] = skel_all
        if self.use_fg_dilation:
            results['dilated_fg'] = dilated_fg
        return results
