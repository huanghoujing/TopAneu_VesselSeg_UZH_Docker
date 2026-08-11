"""
Example usage:
    src_dset_dir=${nnUNet_raw}/Dataset571_TopAneu_Vessel_36fgCls \
    dst_dset_dir=${nnUNet_raw}/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap \
    num_processes=10 \
    left_prefix='L-' \
    right_prefix='R-' \
    python nnunetv2/houjing_scripts/create_left_right_swap_samples.py
"""
import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json

def process_single_case(case_data, images_dir, output_images_dir, left_right_pairs, is_label=False, postfix='_lr_swap'):
    """
    Process flipping and label swapping for a single case
    
    Parameters:
    case_data: Tuple containing case information (image_file, case_name)
    images_dir: Input image directory
    output_images_dir: Output image directory
    left_right_pairs: Dictionary of left-right category pairs
    """
    image_file, case_name = case_data
    
    # Build file path
    image_path = os.path.join(images_dir, image_file)
    
    # Read image and labels
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        print(f"Error: Failed to read file {case_name}: {e}")
        return f"Error: Failed to read file {case_name}: {e}"
    
    # Convert to numpy array numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Perform left-right mirror flip on x-axis (assuming x-axis is the last dimension)
    flipped_image_array = np.flip(image_array, axis=-1)
    
    if is_label:
        flipped_image_array = swap_left_right_labels(flipped_image_array, left_right_pairs)

    # Convert back to SimpleITK imageo SimpleITK image
    flipped_image = sitk.GetImageFromArray(flipped_image_array)
    
    # Copy metadata from original image
    flipped_image.CopyInformation(image)
    
    # Set output filenameput filename
    output_image_file = case_name + postfix + '.nii.gz' if is_label else case_name + postfix + '_0000.nii.gz'
    
    # Save file
    output_image_path = os.path.join(output_images_dir, output_image_file)
    
    sitk.WriteImage(flipped_image, output_image_path)
    
    return f"Successfully processed: {case_name}"

def flip_and_swap_labels_parallel(images_dir, output_images_dir, left_right_pairs, num_processes=10, is_label=False, postfix='_lr_swap'):
    """
    Perform left-right mirror flipping on images and labels in parallel using multiprocessing, and swap left-right category labels
    
    Parameters:
    images_dir: Input image directory
    output_images_dir: Output image directory
    left_right_pairs: Dictionary of left-right category pairs
    num_processes: Number of processes
    """
    
    # Create output directory
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Get all nii.gz files in the images directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    
    print(f"Found {len(image_files)} image files")
    print(f"Processing in parallel using {num_processes} processes")
    
    # Prepare case data
    case_data_list = []
    for image_file in image_files:
        case_name = image_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
        case_data_list.append((image_file, case_name))
    
    # Create partial function, fixing all parameters except case_data
    process_func = partial(
        process_single_case,
        images_dir=images_dir,
        output_images_dir=output_images_dir,
        left_right_pairs=left_right_pairs,
        is_label=is_label,
        postfix=postfix
    )
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, case_data_list),
            total=len(case_data_list),
            desc="Processing cases"
        ))
    
    # Print processing results
    for result in results:
        if result.startswith("Warning") or result.startswith("Error"):
            print(result)
    
    print("Processing complete!")

def swap_left_right_labels(label_array, left_right_pairs):
    """
    Swap left-right category labels
    
    Parameters:
    label_array: Label array
    left_right_pairs: Dictionary of left-right category pairs
    
    Returns:
    Swapped label array
    """
    swapped_array = label_array.copy()
    
    for old_value, new_value in left_right_pairs.items():
        mask = label_array == old_value
        swapped_array[mask] = new_value
    
    return swapped_array

def create_left_right_pairs(labels_dict, left_prefix='Left ', right_prefix='Right '):
    """
    Automatically create left-right pairing relationships based on label dictionary
    
    Parameters:
    labels_dict: Label dictionary containing all labels
    
    Returns:
    Left-right pairing dictionary
    """
    left_right_pairs = {}
    
    # Extract all label names and values
    label_names = list(labels_dict.keys())
    label_values = list(labels_dict.values())
    
    # Find left-right paired labels-right paired labels
    for i, name1 in enumerate(label_names):
        if name1.startswith(right_prefix):
            # Find corresponding Left label
            corresponding_left = name1.replace(right_prefix, left_prefix)
            if corresponding_left in labels_dict:
                right_value = labels_dict[name1]
                left_value = labels_dict[corresponding_left]
                left_right_pairs[right_value] = left_value
                left_right_pairs[left_value] = right_value
                print(f"Found left-right pair: {name1}({right_value}) <-> {corresponding_left}({left_value})")
    
    return left_right_pairs

if __name__ == "__main__":
    if os.environ.get('src_dset_dir') and os.environ.get('dst_dset_dir'):
        src_dset_dir = os.environ['src_dset_dir']
        dst_dset_dir = os.environ['dst_dset_dir']
        print(f"Using environment variables for dataset directories: \n\tsrc_dset_dir={src_dset_dir} \n\tdst_dset_dir={dst_dset_dir}")
    else:
        nnunet_raw = os.environ['nnUNet_raw']  # Make sure to set this environment variable to your nnUNet_raw directory before running the script
        src_dset_dir = os.path.join(nnunet_raw, 'Dataset571_TopAneu_Vessel_36fgCls')
        dst_dset_dir = os.path.join(nnunet_raw, 'Dataset572_TopAneu_Vessel_36fgCls_wLRSwap')
    os.makedirs(dst_dset_dir, exist_ok=True)
    
    imagesTr_dir = os.path.join(src_dset_dir, "imagesTr")
    output_imagesTr_dir = os.path.join(dst_dset_dir, "imagesTr")

    labelsTr_dir = os.path.join(src_dset_dir, "labelsTr")
    output_labelsTr_dir = os.path.join(dst_dset_dir, "labelsTr")
    
    # Label definitions
    labels = json.load(open(os.path.join(src_dset_dir, 'dataset.json')))['labels']
    left_prefix = os.environ.get('left_prefix', 'L-')
    right_prefix = os.environ.get('right_prefix', 'R-')

    # Example Log:
    # Automatically create left-right pairing relationships
    # Found left-right pair: R-P1P2(2) <-> L-P1P2(3)
    # Found left-right pair: R-ICA(4) <-> L-ICA(6)
    # Found left-right pair: R-M1(5) <-> L-M1(7)
    # Found left-right pair: R-Pcom(8) <-> L-Pcom(9)
    # Found left-right pair: R-A1A2(11) <-> L-A1A2(12)
    # Found left-right pair: R-A3(13) <-> L-A3(14)
    # Found left-right pair: R-M2(17) <-> L-M2(19)
    # Found left-right pair: R-M3(18) <-> L-M3(20)
    # Found left-right pair: R-P3P4(21) <-> L-P3P4(22)
    # Found left-right pair: R-VA(23) <-> L-VA(24)
    # Found left-right pair: R-SCA(25) <-> L-SCA(26)
    # Found left-right pair: R-AICA(27) <-> L-AICA(28)
    # Found left-right pair: R-PICA(29) <-> L-PICA(30)
    # Found left-right pair: R-AChA(31) <-> L-AChA(32)
    # Found left-right pair: R-OA(33) <-> L-OA(34)
    # Found left-right pair: R-BVR(38) <-> L-BVR(39)
    # Found left-right pair: R-ECA(41) <-> L-ECA(42)
    # Found left-right pair: R-STA(43) <-> L-STA(44)
    # Found left-right pair: R-MaxA(45) <-> L-MaxA(46)
    # Found left-right pair: R-MMA(47) <-> L-MMA(48)
    # Created left-right pairs: {2: 3, 3: 2, 4: 6, 6: 4, 5: 7, 7: 5, 8: 9, 9: 8, 11: 12, 12: 11, 13: 14, 14: 13, 17: 19, 19: 17, 18: 20, 20: 18, 21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27, 29: 30, 30: 29, 31: 32, 32: 31, 33: 34, 34: 33, 38: 39, 39: 38, 41: 42, 42: 41, 43: 44, 44: 43, 45: 46, 46: 45, 47: 48, 48: 47}
    left_right_pairs = create_left_right_pairs(labels, left_prefix=left_prefix, right_prefix=right_prefix)
    
    print(f"Created left-right pairs: {left_right_pairs}")
    
    # Set number of processes
    num_processes = int(os.environ.get('num_processes', 8))
    print(f"Using num_processes={num_processes} for parallel processing")

    # Copy original imagesTr and labelsTr to the new dataset directory
    shutil.copytree(imagesTr_dir, output_imagesTr_dir, dirs_exist_ok=True)
    print(f"Copied original imagesTr to {output_imagesTr_dir}")
    shutil.copytree(labelsTr_dir, output_labelsTr_dir, dirs_exist_ok=True)
    print(f"Copied original labelsTr to {output_labelsTr_dir}")
    
    # imagesTr
    flip_and_swap_labels_parallel(
        images_dir=imagesTr_dir,
        output_images_dir=output_imagesTr_dir,
        left_right_pairs=left_right_pairs,
        num_processes=num_processes,
        is_label=False,
        postfix='_lr_swap'
    )

    # labelsTr
    flip_and_swap_labels_parallel(
        images_dir=labelsTr_dir,
        output_images_dir=output_labelsTr_dir,
        left_right_pairs=left_right_pairs,
        num_processes=num_processes,
        is_label=True,
        postfix='_lr_swap'
    )

    print(f"Left-right swapped samples created in {dst_dset_dir}")