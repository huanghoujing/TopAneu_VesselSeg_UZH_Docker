import os
import numpy as np
import nibabel as nib
import glob

def save_nifti_w_unit_spacing(array, out_file):
    assert len(array.shape) == 3, f"Invalid shape {array.shape}"
    img = nib.Nifti1Image(array, np.eye(4))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    nib.save(img, out_file)
    print(f"Saved to {out_file}, shape {array.shape}")

def main():
    data_dir = '/home/houjing/Data/nnunet_data/preprocessed/Dataset635_vessel_anatomy_aneurysm_26classes_4595/nnUNetPlans_3d_fullres'
    np_files = glob.glob(f"{data_dir}/*.npz")
    print(f"{len(np_files)} found")

    # np_file = np_files[2000]
    # case_id = os.path.basename(np_file).replace('.npz', '')

    # case_id = '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'
    # np_file = f"{data_dir}/{case_id}.npz"

    case_ids = [
        ('1.2.826.0.1.3680043.8.498.10183727561065274266314159653049375993', 'MRI_T2'),
        ('1.2.826.0.1.3680043.8.498.10207110118916220264491289532161991004', 'MRI_T2'),
        ('1.2.826.0.1.3680043.8.498.32250259987224176174516959348681094310', 'MRI_T2'),
        ('1.2.826.0.1.3680043.8.498.33206606531139273717276688565946361119', 'MRI_T1_Post'),
        ('1.2.826.0.1.3680043.8.498.33295223901007721474389902475960072289', 'MRI_T2'),
        ('1.2.826.0.1.3680043.8.498.33548549469798727567174332201671732647', 'MRI_T2'),
        ('1.2.826.0.1.3680043.8.498.33663859507251796565430018793087839834', 'MRI_T2'),
    ]
    for case_id, modality in case_ids:
        np_file = f"{data_dir}/{case_id}.npz"
        # typical shape [1,27,215]
        loaded = np.load(np_file)
        assert loaded['data'].shape[0] == 1, f"Invalid shape {loaded['data'].shape}"
        assert loaded['seg'].shape[0] == 1, f"Invalid shape {loaded['seg'].shape}"
        save_nifti_w_unit_spacing(loaded['data'][0], f'/home/houjing/Data/tmp/1007_vis_nnunet_img/{modality}/img/{case_id}.nii.gz')
        save_nifti_w_unit_spacing(loaded['seg'][0], f'/home/houjing/Data/tmp/1007_vis_nnunet_img/{modality}/seg/{case_id}.nii.gz')

if __name__ == '__main__':
    main()