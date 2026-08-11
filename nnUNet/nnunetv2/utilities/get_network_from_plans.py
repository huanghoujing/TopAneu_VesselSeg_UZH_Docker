import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


def get_network_from_plans_for_dual_task(
        arch_class_name, arch_kwargs, arch_kwargs_req_import,
        input_channels, output_channels,
        allow_init=True, deep_supervision=None,
        m2f_num_queries=20, m2f_hidden_dim=192, m2f_dec_layers=6,
        m2f_dim_feedforward=None):
    """Build a DualTaskResEncUNet using the same plan kwargs as nnUNet.

    Extracts the encoder/decoder kwargs from plans and passes them to
    DualTaskResEncUNet together with Mask2Former head parameters.
    """
    from nnunetv2.model.dual_task_unet import DualTaskResEncUNet

    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    if deep_supervision is not None and 'deep_supervision' not in architecture_kwargs:
        architecture_kwargs['deep_supervision'] = deep_supervision

    architecture_kwargs['m2f_num_queries'] = m2f_num_queries
    architecture_kwargs['m2f_hidden_dim'] = m2f_hidden_dim
    architecture_kwargs['m2f_dec_layers'] = m2f_dec_layers
    architecture_kwargs['m2f_dim_feedforward'] = m2f_dim_feedforward

    network = DualTaskResEncUNet(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs,
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network
