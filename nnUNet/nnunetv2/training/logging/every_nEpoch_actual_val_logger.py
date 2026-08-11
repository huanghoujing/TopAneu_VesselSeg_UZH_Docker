import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
from collections import defaultdict

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class MetricLogger(object):
    """
    Removed Keys: ema_fg_dice, val_losses.
    Added Keys: mean_dice.
    Allow saving validation metrics at sparse epochs. Use `val_at_epochs` to record these epoch numbers.
    Change value of `dice_per_class_or_region` to dict.
    Removed Plot: epoch times
    Added Plot: dice_per_class_or_region
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'dice_per_class_or_region': defaultdict(list),
            'val_at_epochs': list(),
            'train_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.val_keys = {'mean_fg_dice', 'dice_per_class_or_region'}
        self.verbose = verbose

    def log(self, key, value, epoch: int):
        assert key in self.my_fantastic_logging.keys(), f"Unexpected key {key}. Available ones:  {list(self.my_fantastic_logging.keys())}"

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if key == 'dice_per_class_or_region':
            for k, v in value.items():
                self.my_fantastic_logging[key][k].append(v)
        else:
            self.my_fantastic_logging[key].append(value)
        if key in self.val_keys:
            if not self.my_fantastic_logging['val_at_epochs'] or self.my_fantastic_logging['val_at_epochs'][-1] < epoch:
                self.my_fantastic_logging['val_at_epochs'].append(epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax2.plot(self.my_fantastic_logging['val_at_epochs'], self.my_fantastic_logging['mean_fg_dice'], color='g', ls='-', label="dice fg", linewidth=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # dice_per_class_or_region
        colors = list(mcolors._colors_full_map.values())
        ax = ax_all[1]
        for (k, _list), c in zip(self.my_fantastic_logging['dice_per_class_or_region'].items(), colors):
            ax.plot(self.my_fantastic_logging['val_at_epochs'], _list, color=c, ls='-', label=str(k), linewidth=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("dice")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
