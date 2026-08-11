from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


raw_label_mapping = {
    "background": 0,
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
    "Upper Right Central Incisor": 11,
    "Upper Right Lateral Incisor": 12,
    "Upper Right Canine": 13,
    "Upper Right First Premolar": 14,
    "Upper Right Second Premolar": 15,
    "Upper Right First Molar": 16,
    "Upper Right Second Molar": 17,
    "Upper Right Third Molar (Wisdom Tooth)": 18,
    "Upper Left Central Incisor": 21,
    "Upper Left Lateral Incisor": 22,
    "Upper Left Canine": 23,
    "Upper Left First Premolar": 24,
    "Upper Left Second Premolar": 25,
    "Upper Left First Molar": 26,
    "Upper Left Second Molar": 27,
    "Upper Left Third Molar (Wisdom Tooth)": 28,
    "Lower Left Central Incisor": 31,
    "Lower Left Lateral Incisor": 32,
    "Lower Left Canine": 33,
    "Lower Left First Premolar": 34,
    "Lower Left Second Premolar": 35,
    "Lower Left First Molar": 36,
    "Lower Left Second Molar": 37,
    "Lower Left Third Molar (Wisdom Tooth)": 38,
    "Lower Right Central Incisor": 41,
    "Lower Right Lateral Incisor": 42,
    "Lower Right Canine": 43,
    "Lower Right First Premolar": 44,
    "Lower Right Second Premolar": 45,
    "Lower Right First Molar": 46,
    "Lower Right Second Molar": 47,
    "Lower Right Third Molar (Wisdom Tooth)": 48,
    "Left Mandibular Incisive Canal": 103,
    "Right Mandibular Incisive Canal": 104,
    "Lingual Canal": 105,
    "Upper Right Central Incisor Pulp": 111,
    "Upper Right Lateral Incisor Pulp": 112,
    "Upper Right Canine Pulp": 113,
    "Upper Right First Premolar Pulp": 114,
    "Upper Right Second Premolar Pulp": 115,
    "Upper Right First Molar Pulp": 116,
    "Upper Right Second Molar Pulp": 117,
    "Upper Right Third Molar (Wisdom Tooth) Pulp": 118,
    "Upper Left Central Incisor Pulp": 121,
    "Upper Left Lateral Incisor Pulp": 122,
    "Upper Left Canine Pulp": 123,
    "Upper Left First Premolar Pulp": 124,
    "Upper Left Second Premolar Pulp": 125,
    "Upper Left First Molar Pulp": 126,
    "Upper Left Second Molar Pulp": 127,
    "Upper Left Third Molar (Wisdom Tooth) Pulp": 128,
    "Lower Left Central Incisor Pulp": 131,
    "Lower Left Lateral Incisor Pulp": 132,
    "Lower Left Canine Pulp": 133,
    "Lower Left First Premolar Pulp": 134,
    "Lower Left Second Premolar Pulp": 135,
    "Lower Left First Molar Pulp": 136,
    "Lower Left Second Molar Pulp": 137,
    "Lower Left Third Molar (Wisdom Tooth) Pulp": 138,
    "Lower Right Central Incisor Pulp": 141,
    "Lower Right Lateral Incisor Pulp": 142,
    "Lower Right Canine Pulp": 143,
    "Lower Right First Premolar Pulp": 144,
    "Lower Right Second Premolar Pulp": 145,
    "Lower Right First Molar Pulp": 146,
    "Lower Right Second Molar Pulp": 147,
    "Lower Right Third Molar (Wisdom Tooth) Pulp": 148
  }


def mapping_DS101() -> Dict[int, int]:
    """Keep all 77 Classes"""    
    unique_labels = sorted(np.unique(list(raw_label_mapping.values())).tolist())
    mapping = {int(label): i for i, label in enumerate(unique_labels)}
    return mapping

def load_json(json_file: str) -> Any:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json(json_file: str, data: Any, indent: int = 4) -> None:
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)


def image_to_nifi(input_path: str, output_path: str) -> None:
    image_sitk = sitk.ReadImage(input_path)
    sitk.WriteImage(image_sitk, output_path)


def label_mapping(input_path: str, output_path: str, mapping: Dict[int, int] = None) -> None:

    label_sitk = sitk.ReadImage(input_path)
    if mapping is not None:
        label_np = sitk.GetArrayFromImage(label_sitk)

        label_np_new = np.zeros_like(label_np, dtype=np.uint8)
        for org_id, new_id in mapping.items():
            label_np_new[label_np == org_id] = new_id

        label_sitk_new = sitk.GetImageFromArray(label_np_new)
        label_sitk_new.CopyInformation(label_sitk)
        sitk.WriteImage(label_sitk_new, output_path)
    else:
        sitk.WriteImage(label_sitk, output_path)


def process_labels(
    files: str, lbl_dir_in: str, lbl_dir_out: str, mapping: Dict[int, int], n_processes: int = 12
) -> None:

    os.makedirs(lbl_dir_out, exist_ok=True)

    iterable = [
        {
            "input_path": join(lbl_dir_in, file),
            "output_path": join(lbl_dir_out, file.replace(".mha", ".nii.gz")),
            "mapping": mapping,
        }
        for file in files
    ]
    with multiprocessing.Pool(processes=n_processes) as pool:
        jobs = [pool.apply_async(label_mapping, kwds={**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc="Process Labels...")]


def process_ds(
    root: str, input_ds: str, output_ds: str, mapping: dict,
) -> None:
    os.makedirs(join(root, output_ds), exist_ok=True)
    os.makedirs(join(root, output_ds, "labelsTr"), exist_ok=True)
    # --- Handle Labels --- #
    lbl_files = os.listdir(join(root, input_ds, "labelsTr"))
    lbl_dir_in = join(root, input_ds, "labelsTr")
    lbl_dir_out = join(root, output_ds, "labelsTr")

    process_labels(lbl_files, lbl_dir_in, lbl_dir_out, mapping, n_processes=12)

    # --- Handle Images --- #
    img_files = os.listdir(join(root, input_ds, "imagesTr"))    
    img_names = [file.replace("_0000.nii.gz", "") for file in img_files]
    
    # --- Generate nnUNet dataset.json --- #
    dataset_json = load_json(join(root, input_ds, "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = output_ds
    dataset_json["numTraining"] = len(lbl_files)
    
    label_dict = dataset_json["labels"]
    label_dict_new = {"background": 0}
    for k, v in label_dict.items():
        if v in mapping.keys():
            label_dict_new[k] = mapping[v]
    dataset_json["labels"] = label_dict_new
    write_json(join(root, output_ds, "dataset.json"), dataset_json)

    # --- Generate nnUNet splits_final.json --- #
    random_seed = 42
    random.seed(random_seed)
    random.shuffle(img_names)

    split_index = int(len(img_names) * 0.8)  # 80:20 split
    train_files = img_names[:split_index]
    val_files = img_names[split_index:]
    train_files.sort()
    val_files.sort()

    split = [{"train": train_files, "val": val_files}]
    write_json(join(root, output_ds, "splits_final.json"), split)

    monai_split_dict = {
        'training': [
            {
                'image': join('imagesTr', f"{name}_0000.nii.gz"),
                'label': join('labelsTr', f"{name}.nii.gz")                
            }
            for name in train_files
        ],
        'validation': [
            {
                'image': join('imagesTr', f"{name}_0000.nii.gz"),
                'label': join('labelsTr', f"{name}.nii.gz")                
            }
            for name in val_files
        ]
    }
    write_json(join(root, output_ds, "monai_splits_0.json"), monai_split_dict)


if __name__ == "__main__":
    process_ds(
        root="/home/houjing/Data",
        input_ds="ToothFairy3",
        output_ds="nnunet_data/raw/Dataset101_ToothFairy3_All",
        mapping=mapping_DS101(),
    )
    