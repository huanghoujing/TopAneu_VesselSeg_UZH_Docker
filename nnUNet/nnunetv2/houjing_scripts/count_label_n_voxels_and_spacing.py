"""
python3 scripts/data_process/count_RSNA_label_n_voxels_and_spacing.py \
    --input_dir /mnt/WD_8T_SSD/data2/nnunet_data/raw/Dataset635_vessel_anatomy_aneurysm_26classes_4595/labelsTr \
    --output_csv_file /mnt/WD_8T_SSD/data2/nnunet_data/raw/Dataset635_vessel_anatomy_aneurysm_26classes_4595/size_info_train_RAS_0p5.csv \
    --ref_spacing 0.5 0.5 0.5 \
    --max_label_value 26 \
    --num_workers 32
"""
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from glob import glob


def _workon_1case(seg_file, casename, max_label_value, ref_spacing=None):
    label_dict = {}
    fg_label_values = list(range(1, max_label_value+1))
    available_label_value_set = set(fg_label_values)
    available_label_value_set.add(0)

    seg = sitk.ReadImage(seg_file)
    seg = sitk.DICOMOrient(seg, 'RAS')
    seg_array = sitk.GetArrayFromImage(seg)
    unique_seg_values = np.unique(seg_array)
    if not set(unique_seg_values).issubset(available_label_value_set):
        print(f"Contain invalid seg label. Unique values: {unique_seg_values}")
        return
    spacing = seg.GetSpacing()
    shape = sitk.GetArrayFromImage(seg).shape
    assert len(shape) == 3, f"Invalid shape {shape}"
    volume_1e6_mm3 = np.prod(shape) * np.prod(spacing) / 10**6
    size_info = {'casename': casename, 'volume_1e6_mm3': volume_1e6_mm3,  'H': shape[0], 'W': shape[1], 'D': shape[2], 'dH': round(spacing[0], 2), 'dW': round(spacing[1], 2), 'dD': round(spacing[2], 2)}
    for lv in fg_label_values:
        label_name = label_dict.get(lv, f"Label_{lv}")
        # voxel size
        voxel_size = np.prod(spacing)
        # number of positive voxels
        num_positive_voxels = np.sum(seg_array == lv)
        # number of voxels if using ref_spacing
        num_voxels_ref = num_positive_voxels if ref_spacing is None else \
            num_positive_voxels * np.prod(spacing) / np.prod(ref_spacing)
        num_voxels_ref = int(num_voxels_ref)
        size_info[label_name] = num_voxels_ref
    print(size_info)
    return size_info

def _find_all_medical_image_files(input_dir):
    """
    Find all medical image files in the input directory and its subdirectories.
    """
    # Patterns to match
    patterns = ['**/*.nii.gz', '**/*.nii', '**/*.nrrd']

    # Collect all matched files
    all_files = []
    for pattern in patterns:
        all_files.extend(glob(os.path.join(input_dir, pattern), recursive=True))
    
    all_files = sorted(all_files)
    print(f"Sorted files [:10]:")
    print('\n'.join(all_files[:10]))  # Print first 10 files for brevity
    return all_files

def _get_casename_from_path(path):
    return os.path.basename(path).replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '')

def save_histogram(array, output_img_path, xlabel, title, bins=100):
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

    plt.figure(figsize=(8, 4))

    counts, bin_edges, _ = plt.hist(array, bins=bins, density=False, color='darkorange', edgecolor='black')
    proportions = counts / counts.sum()
    plt.clf()
    plt.bar( (bin_edges[:-1] + bin_edges[1:]) / 2, proportions, width=np.diff(bin_edges), color='darkorange', edgecolor='black' )

    # plt.hist(array, bins=bins, density=True, color='darkorange', edgecolor='black')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    print(f"[*] Saved {output_img_path}")

def main_in_order():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--max_label_value", type=int, required=True)
    parser.add_argument("--output_csv_file", type=str, required=True)
    parser.add_argument("--ref_spacing", type=float, nargs='+', required=False, help="A list of reference RAS spacing for resampling. Default is None.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers to use for processing.")
    args = parser.parse_args()

    if True:
        os.makedirs(os.path.dirname(args.output_csv_file), exist_ok=True)
        
        seg_files = _find_all_medical_image_files(args.input_dir)

        # Process files in parallel using ProcessPoolExecutor.
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit futures in order and keep a mapping to indices
            futures = []
            for seg_file in seg_files:
                future = executor.submit(_workon_1case, seg_file, _get_casename_from_path(seg_file), args.max_label_value, args.ref_spacing)
                futures.append(future)
            
            # Wait for all to complete
            results = [None] * len(futures)
            for i, future in enumerate(tqdm(futures, desc="Process Files")):
                try:
                    res = future.result()
                    if res is not None:
                        results[i] = res
                except Exception as exc:
                    print(f"Error processing {seg_files[i]}: {traceback.format_exc()}")

        # Save results to a CSV file
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv_file, index=False)
        print(f"Size information saved to {args.output_csv_file}")

        print("All processing completed.")

    ####################################################
    # Parse CSV and Save histograms
    ####################################################

    if False:
        hist_dir = os.path.join( os.path.dirname(args.output_csv_file), 'histograms' )
        df = pd.read_csv(args.output_csv_file)
        
        save_histogram(df['volume_1e6_mm3'].to_numpy(), f"{hist_dir}/volume_1e6_mm3.png", xlabel='volume size / 1e6 mm3', title=f'Volume Size Distribution', bins=100)
        save_histogram(df['dH'].to_numpy(), f"{hist_dir}/spacing_H.png", xlabel='Spacing H', title=f'Distribution of Spacing H', bins=100)
        save_histogram(df['dW'].to_numpy(), f"{hist_dir}/spacing_W.png", xlabel='Spacing W', title=f'Distribution of Spacing W', bins=100)
        save_histogram(df['dD'].to_numpy(), f"{hist_dir}/spacing_D.png", xlabel='Spacing D', title=f'Distribution of Spacing D', bins=100)
        for label_value in range(1, args.max_label_value+1):
            col_name = f"Label_{label_value}"
            save_histogram(df[col_name].to_numpy(), f"{hist_dir}/{col_name}.png", xlabel='NO. Voxels', title=f'Size Distribution of {col_name}', bins=100)
    
    ####################################################
    # Parse CSV and Count number of positive samples for each label
    ####################################################

    if False:
        out_dir = os.path.dirname(args.output_csv_file)
        df = pd.read_csv(args.output_csv_file)
        col_names = [f"Label_{label_value}" for label_value in range(1, args.max_label_value+1)]
        
        this_df = df[df['dD'] < 1.5]
        out_csv = f"{out_dir}/thin_slice_samples.csv"
        this_df.to_csv(out_csv, index=False)
        print(f"[*] Saved {out_csv}, shape {this_df.shape}")
        this_df = (this_df[col_names] != 0).sum().to_frame().T
        out_csv = f"{out_dir}/thin_slice_n_positive_samples_per_label.csv"
        this_df.to_csv(out_csv, index=False)
        print(f"[*] Saved {out_csv}, shape {this_df.shape}")

        this_df = df[df['dD'] >= 1.5]
        out_csv = f"{out_dir}/thick_slice_samples.csv"
        this_df.to_csv(out_csv, index=False)
        print(f"[*] Saved {out_csv}, shape {this_df.shape}")
        this_df = (this_df[col_names] != 0).sum().to_frame().T
        out_csv = f"{out_dir}/thick_slice_n_positive_samples_per_label.csv"
        this_df.to_csv(out_csv, index=False)
        print(f"[*] Saved {out_csv}, shape {this_df.shape}")

    
if __name__ == "__main__":
    main_in_order()