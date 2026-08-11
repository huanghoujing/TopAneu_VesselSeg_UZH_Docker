import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3DNoCrop(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = []
        seg_all = []
        case_properties = []

        for j, i in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)
            data_all.append(data)
            seg_all.append(seg)

        if self.transforms is not None:
            if torch is not None:
                torch_nthreads = torch.get_num_threads()
                torch.set_num_threads(1)
            with threadpool_limits(limits=1, user_api=None):
                images = []
                segs = []
                for b in range(self.batch_size):
                    # Copy arrays to make them writable - memory-mapped arrays are read-only
                    # and transforms may modify tensors in-place, causing segfaults
                    tmp = self.transforms(**{'image': torch.from_numpy(data_all[b].copy()).float(), 'segmentation': torch.from_numpy(seg_all[b].copy()).to(torch.int16)})
                    images.append(tmp['image'])
                    segs.append(tmp['segmentation'])
                data_all = torch.stack(images)
                
                # HOUJING: BUG
                # seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                # https://github.com/MIC-DKFZ/nnUNet/blob/f1851fbaf2c53dcb51b079b60a01de528a7d0c17/nnunetv2/training/dataloading/data_loader.py#L211
                if isinstance(segs[0], list):
                    seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                else:
                    seg_all = torch.stack(segs)

                del segs, images
            if torch is not None:
                torch.set_num_threads(torch_nthreads)
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
