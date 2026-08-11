import numpy as np

def create_ce_loss_w_list(dice_per_cls):
    """Create a list of CE loss weights based on per-class Dice scores.
    dice_per_cls: List of Dice scores for each foreground class (excluding background).
    Returns a comma-separated string representation, no spaces.
    """
    x = np.array(dice_per_cls)
    print(f"{len(x)} FG classes, dice_per_cls: {x}")
    # print the classes with dice < threshold, 0.1, 0.2, ..., 0.9
    for i in range(1, 10):
        thresh = i * 0.1
        classes = [j+1 for j in range(len(x)) if x[j] <= thresh]
        print(f"Classes with dice <= {thresh:<4.1f}, {len(classes)} classes: {classes}")
    # if dice <= 0.3, set w = 2.5
    # if dice <= 0.5, set w = 2
    # if dice <= 0.7, set w = 1.5
    # else, set w = 1
    ce_loss_w_list = [1]  # background class weight is always 1
    for dice in dice_per_cls:
        if dice <= 0.3:
            ce_loss_w_list.append(2.5)
        elif dice <= 0.5:
            ce_loss_w_list.append(2)
        elif dice <= 0.7:
            ce_loss_w_list.append(1.5)
        else:
            ce_loss_w_list.append(1)
    w_list_str = ','.join([f"{w:.1f}" for w in ce_loss_w_list])
    print(f"CE loss weight list: {w_list_str}")
    return w_list_str

if __name__ == "__main__":
    # The DICEs from a model without applying ce_class_weight
    ce_loss_w_list_str = create_ce_loss_w_list(
        [np.float64(0.9481), np.float64(0.9075), np.float64(0.9052), np.float64(0.9379), np.float64(0.9173), np.float64(0.9389), np.float64(0.9206), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.9014), np.float64(0.8965), np.float64(0.8232), np.float64(0.6752), np.float64(0.0), np.float64(0.0), np.float64(0.8701), np.float64(0.7617), np.float64(0.8695), np.float64(0.742), np.float64(0.5649), np.float64(0.7064), np.float64(0.9279), np.float64(0.9237), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.9504), np.float64(0.9514)]
    )
    # After applying ce_class_weight, the DICEs of the same model are improved, especially for those classes with 0 DICEs before.
    # ce_loss_w_list_str = create_ce_loss_w_list(
    #     [np.float64(0.9495), np.float64(0.899), np.float64(0.9068), np.float64(0.9348), np.float64(0.9076), np.float64(0.8405), np.float64(0.9187), np.float64(0.8579), np.float64(0.2634), np.float64(0.7523), np.float64(0.901), np.float64(0.912), np.float64(0.8522), np.float64(0.6971), np.float64(0.437), np.float64(0.1446), np.float64(0.7709), np.float64(0.6102), np.float64(0.7752), np.float64(0.6699), np.float64(0.5747), np.float64(0.7674), np.float64(0.9298), np.float64(0.9174), np.float64(0.7337), np.float64(0.716), np.float64(0.5291), np.float64(0.101), np.float64(0.622), np.float64(0.829), np.float64(0.2037), np.float64(0.2478), np.float64(0.7825), np.float64(0.691), np.float64(0.9457), np.float64(0.8539)]
    # )
