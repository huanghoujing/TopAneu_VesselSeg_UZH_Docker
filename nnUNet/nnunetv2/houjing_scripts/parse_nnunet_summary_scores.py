import json
import numpy as np


def _parse_topBrainPos(in_json_file):
    print(f"Parsing {in_json_file}...")
    label_names = [str(i) for i in range(1, 14)]
    
    with open(in_json_file, 'r') as f:
        data = json.load(f)
    
    all_scores = {name: round(data['mean'][name]['Dice'], 3) for name in label_names}
    print(all_scores)
    
    label_names.remove('9')  # some model has NaN or 0 score for this label
    selected_scores = [all_scores[name] for name in label_names]
    mean_score = np.mean(selected_scores)
    std_score = np.std(selected_scores)
    print(f"\nScores averaged over {len(label_names)} labels: {label_names}\n\t{mean_score:.3f} +- {std_score:.3f}")

def main():
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/nnUNetTrainerEp300NoLRMirrorNoDeepSupwTopK__nnUNetPlans__3d_fullres/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/nnUNetTrainerEp300NoLRMirrorNoDeepSupwTopK__nnUNetPlans__3d_fullres_fromTopCoW/fold_0/validation/summary.json')

    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM__nnUNetPlans__3d_fullres/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3__nnUNetPlans__3d_fullres/fold_0/validation/summary.json')
    
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_overFG0p3__nnUNetPlans__3d_fullres/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3__nnUNetPlans__3d_fullres_even_spacing/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3__nnUNetPlans__3d_fullres_even_spacing/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3__nnUNetPlans__3d_fullres_even_spacing_0p6/fold_0/validation/summary.json')
    # _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3__nnUNetPlans__3d_fullres_even_spacing_0p6_noBatchDice/fold_0/validation/summary.json')
    _parse_topBrainPos('/home/houjing/Data/nnunet_data/results/Dataset510_topbrain_POS/TopBrainPos_Tr_rot30_DiffClusterSM_overFG0p3_cowPT_LRx0p1__nnUNetPlans__3d_fullres_even_spacing_0p6_noBatchDice_ps128x160x160/fold_0/validation/summary.json')


if __name__ == "__main__":
    main()