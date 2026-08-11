"""Modify checkpoint file attributes.
Example:

python nnunetv2/houjing_scripts/modify_ckpt_attr.py \
    --ckpt_file xxxx/checkpoint_final.pth \
    --func_name set_default_trainer_rm_mirror

python nnunetv2/houjing_scripts/modify_ckpt_attr.py \
    --ckpt_file xxxx/checkpoint_final.pth \
    --new_ckpt_file xxxx/checkpoint_final_default_name_wo_optim.pth \
    --func_name set_default_trainer_rm_optim

python nnunetv2/houjing_scripts/modify_ckpt_attr.py \
    --ckpt_file xxxx/checkpoint_final.pth \
    --new_ckpt_file xxxx/checkpoint_final_wo_optim.pth \
    --func_name rm_network_keys_w_substr_and_optim
    
"""
import argparse
import json
import torch


def modify(ckpt_file, new_ckpt_file, **kwargs):
    if new_ckpt_file is None:
        new_ckpt_file = ckpt_file
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'), weights_only=False)
    for k, v in kwargs.items():
        old_v = checkpoint.get(k, 'MISSING')
        checkpoint[k] = v
        print(f"KEY: {k}, VALUE: {old_v} -> {v}")
    torch.save(checkpoint, new_ckpt_file)
    print(f"Modify checkpoint file {ckpt_file} to {new_ckpt_file} successfully!")

def set_default_trainer_rm_mirror(ckpt_file, new_ckpt_file):
    modify(
        ckpt_file, 
        new_ckpt_file,
        trainer_name='nnUNetTrainer',
        inference_allowed_mirroring_axes=None
    )

def set_default_trainer_rm_optim(ckpt_file, new_ckpt_file):
    """These are what nnUNet saved:
        checkpoint = {
            'network_weights': mod.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'logging': self.logger.get_checkpoint(),
            '_best_ema': self._best_ema,
            'current_epoch': self.current_epoch + 1,
            'init_args': self.my_init_kwargs,
            'trainer_name': self.__class__.__name__,
            'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
        }
    """
    if new_ckpt_file is None:
        new_ckpt_file = ckpt_file
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'), weights_only=False)
    checkpoint.pop('optimizer_state')
    checkpoint['trainer_name'] = 'nnUNetTrainer'
    torch.save(checkpoint, new_ckpt_file)
    print(f"Modify checkpoint file {ckpt_file} to {new_ckpt_file} successfully!")

def rm_network_keys_w_substr_and_optim(ckpt_file, new_ckpt_file, substr=[]):
    print(f"Removing network keys with substring: {substr}")
    if new_ckpt_file is None:
        new_ckpt_file = ckpt_file
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(substr, str):
        substr = [substr]
    keys_to_remove = []
    for s in substr:
        keys_to_remove += [k for k in checkpoint['network_weights'].keys() if s in k]
    for k in keys_to_remove:
        checkpoint['network_weights'].pop(k, None)
        print(f"Removed key: {k}")
    if 'optimizer_state' in checkpoint:
        checkpoint.pop('optimizer_state', None)
        print("Removed optimizer_state from checkpoint.")
    torch.save(checkpoint, new_ckpt_file)
    print(f"Modify checkpoint file {ckpt_file} to {new_ckpt_file} successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--new_ckpt_file", type=str, required=False, default=None)
    parser.add_argument("--func_name", type=str, required=False, default='modify')
    parser.add_argument("--kwargs", type=json.loads, required=False, default={})
    args = parser.parse_args()
    print("Arguments:")
    print(args)
    if args.func_name == 'modify':
        modify(args.ckpt_file, args.new_ckpt_file, **args.kwargs)
    else:
        globals()[args.func_name](args.ckpt_file, args.new_ckpt_file, **args.kwargs)


if __name__ == "__main__":
    main()