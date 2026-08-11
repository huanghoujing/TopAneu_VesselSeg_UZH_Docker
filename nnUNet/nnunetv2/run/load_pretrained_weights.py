from collections.abc import Mapping

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback


def _load_pretrained_checkpoint(fname):
    """Load source weights on CPU before copying compatible tensors to the target model."""
    return torch.load(fname, map_location='cpu', weights_only=False)


def extract_pretrained_weights(saved_model: Mapping):
    """Return a network state dict from an nnU-Net or VesselFM checkpoint.

    nnU-Net checkpoints store network tensors in ``network_weights``. VesselFM
    checkpoints are PyTorch Lightning checkpoints whose segmentation network
    uses ``state_dict`` keys prefixed by ``model.seg_net.``. The primary model
    is used in preference to the auxiliary EMA copy, which is only a fallback.
    """
    if not isinstance(saved_model, Mapping):
        raise TypeError(
            f"Expected a mapping-like checkpoint, got {type(saved_model).__name__}."
        )

    if 'network_weights' in saved_model:
        pretrained_dict = saved_model['network_weights']
        source_description = "nnU-Net checkpoint key 'network_weights'"
    elif 'state_dict' in saved_model:
        state_dict = saved_model['state_dict']
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Checkpoint key 'state_dict' must contain a mapping of parameter names to tensors."
            )

        pretrained_dict = None
        source_description = None
        for prefix in ('model.seg_net.', 'ema_model.seg_net.'):
            matched_weights = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if matched_weights:
                pretrained_dict = matched_weights
                source_description = f"PyTorch Lightning VesselFM state_dict prefix '{prefix}'"
                break

        if pretrained_dict is None:
            state_dict_keys = list(state_dict)[:10]
            raise KeyError(
                "The PyTorch Lightning checkpoint does not contain VesselFM segmentation weights under "
                "'model.seg_net.' or 'ema_model.seg_net.'. "
                f"First state_dict keys: {state_dict_keys}"
            )
    else:
        checkpoint_keys = list(saved_model)[:20]
        raise KeyError(
            "Unsupported pretrained checkpoint format. Expected an nnU-Net 'network_weights' key or a "
            f"VesselFM PyTorch Lightning 'state_dict' key. Top-level keys: {checkpoint_keys}"
        )

    if not isinstance(pretrained_dict, Mapping):
        raise TypeError(
            f"{source_description} must contain a mapping of parameter names to tensors."
        )
    if not pretrained_dict:
        raise ValueError(f"{source_description} did not contain any network weights.")

    return pretrained_dict, source_description


def print_ckpt_info(loaded_ckpt):
    keys = ['current_epoch', 'epoch', 'global_step', '_best_ema', 'trainer_name']
    print("Checkpoint Info:")
    for key in keys:
        if key in loaded_ckpt:
            print(f"\t{key}: {loaded_ckpt[key]}")

def load_pretrained_weights(network, fname, verbose=False, skip_seg_layers=True):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if network is None:
        raise ValueError("You need to provide a network to load the pretrained weights into.")
    
    saved_model = _load_pretrained_checkpoint(fname)
    pretrained_dict, source_description = extract_pretrained_weights(saved_model)

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ] if skip_seg_layers else []

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        print(f"[ ] Detected that the model is a torch.dynamo OptimizedModule. Extracting the original module.")
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    print_ckpt_info(saved_model)
    print(f"Using {source_description} ({len(pretrained_dict)} tensors).")
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


def load_pretrained_weights_hjh(network, fname, verbose=False, strict=False, skip_seg_layers=False):
    if network is None:
        raise ValueError("You need to provide a network to load the pretrained weights into.")
    
    saved_model = _load_pretrained_checkpoint(fname)
    pretrained_dict, source_description = extract_pretrained_weights(saved_model)

    print("################### Loading pretrained weights from file ", fname, '###################')
    print_ckpt_info(saved_model)
    print(f"Using {source_description} ({len(pretrained_dict)} tensors).")

    if skip_seg_layers:
        n_params = len(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if '.seg_layers.' not in k}
        print(f"Skipping loading of segmentation layers. Number of parameters after filtering: {n_params} -> {len(pretrained_dict)}")

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        print(f"[ ] Detected that the model is a torch.dynamo OptimizedModule. Extracting the original module.")
        mod = mod._orig_mod

    if not strict:
        model_dict = mod.state_dict()
        shape_mismatch = [
            k for k in pretrained_dict
            if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape
        ]
        if shape_mismatch:
            print(f"Filtering {len(shape_mismatch)} keys with shape mismatch (strict=False):")
            for k in shape_mismatch:
                print(f"\t{k}: ckpt {list(pretrained_dict[k].shape)} vs model {list(model_dict[k].shape)}")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in shape_mismatch}

    incompatible_keys = mod.load_state_dict(pretrained_dict, strict=strict)

    missing_str = '\n\t'.join(incompatible_keys.missing_keys)
    unexpected_str = '\n\t'.join(incompatible_keys.unexpected_keys)
    print(f"missing_keys:\n\t{missing_str}")
    print(f"unexpected_keys:\n\t{unexpected_str}")
    print("################### Done ###################")

def load_pretrained_weights_w_postnet_hjh(network, post_net, fname, verbose=False, strict=False, skip_seg_layers=False):
    if network is None:
        raise ValueError("You need to provide a network to load the pretrained weights into.")
    
    saved_model = _load_pretrained_checkpoint(fname)
    pretrained_dict, source_description = extract_pretrained_weights(saved_model)

    print("################### Loading pretrained weights from file ", fname, '###################')
    print_ckpt_info(saved_model)
    print(f"Using {source_description} ({len(pretrained_dict)} tensors).")

    if skip_seg_layers:
        n_params = len(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if '.seg_layers.' not in k}
        print(f"Skipping loading of segmentation layers. Number of parameters after filtering: {n_params} -> {len(pretrained_dict)}")

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        print(f"[ ] Detected that the model is a torch.dynamo OptimizedModule. Extracting the original module.")
        mod = mod._orig_mod
    
    incompatible_keys = mod.load_state_dict(pretrained_dict, strict=strict)

    missing_str = '\n\t'.join(incompatible_keys.missing_keys)
    unexpected_str = '\n\t'.join(incompatible_keys.unexpected_keys)
    print(f"missing_keys:\n\t{missing_str}")
    print(f"unexpected_keys:\n\t{unexpected_str}")
    print("################### Done ###################")

    post_net.load_state_dict(saved_model['post_net_weights'])
    print("################### Loaded post processing net weights ###################")
