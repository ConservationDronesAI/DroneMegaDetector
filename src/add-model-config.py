#%% Header

"""
add-model-config.py

Re-save an RF-DETR checkpoint with the 'model_config' metadata that rfdetr (since
v1.8.0) writes into the checkpoints it produces, and that rfdetr.from_checkpoint()
uses to rebuild the architecture the model was trained with.

This script records exactly how DroneMegaDetector_RFDETRMedium_2026.10.04.pth was
produced from DroneMegaDetector_RFDETRMedium_2026.10.01.pth.

The original checkpoint was written by a pre-release build of rfdetr (it reports
rfdetr_version 1.6.0), in the layout rfdetr 1.7.x uses for "best" checkpoints: it
records model_name ('RFDETRMedium') and the training args, but no model_config.
Without model_config, from_checkpoint() builds the model at the RFDETRMedium default
resolution (576) rather than the resolution the model was trained at (1280), and
interpolates the trained 80x80 positional embeddings down to 36x36.

This script makes the following changes, and nothing else:

- Adds a top-level 'model_config' dict: the default config for the checkpoint's
  model_name, with the training resolution and num_classes (inferred from the
  classification head).  positional_encoding_size follows from the resolution
  (1280 / 16 = 80).  'device' and 'pretrain_weights' are omitted, since
  from_checkpoint() ignores both.

- Adds the 'validate_loop' and 'test_loop' stubs to 'loops', which rfdetr 1.11's
  strip_checkpoint() backfills into older checkpoints.

- Sets 'rfdetr_version' to the version of rfdetr installed when this script runs.

The model weights, 'args' (including class names), 'model_name', and the PyTorch
Lightning bookkeeping keys are copied unchanged.  'model' and 'state_dict' continue
to share storage, so the weights are still only stored once.

Before writing anything, the new model_config is checked against the weights
(positional embedding grid, decoder depth, query count) and against the
architecture fields recorded in 'args'.
"""

#%% Imports and constants

import argparse
import math
import os
import re
import sys

from importlib.metadata import version

import torch
import rfdetr

# The resolution DroneMegaDetector was trained at
DEFAULT_RESOLUTION = 1280


#%% Support functions

def _check_model_config(model_config, checkpoint):
    """
    Verify that [model_config] is consistent with the weights and training args
    in [checkpoint], raising a ValueError if it isn't.

    Args:
        model_config (dict): the model config to be written
        checkpoint (dict): the loaded checkpoint
    """

    weights = checkpoint['model']
    args = checkpoint['args']
    errors = []

    # The positional embedding has one entry per patch on a square grid, plus a
    # class token
    pos_embed_keys = [k for k in weights if k.endswith('embeddings.position_embeddings')]
    for k in pos_embed_keys:
        grid_size = math.isqrt(weights[k].shape[1] - 1)
        if grid_size != model_config['positional_encoding_size']:
            errors.append(f'{k} has a {grid_size}x{grid_size} grid, but '
                          f'positional_encoding_size is {model_config["positional_encoding_size"]}')

    decoder_layers = {int(m.group(1)) for k in weights
                      for m in [re.match(r'transformer\.decoder\.layers\.(\d+)\.', k)] if m}
    if len(decoder_layers) != model_config['dec_layers']:
        errors.append(f'weights have {len(decoder_layers)} decoder layers, but '
                      f'dec_layers is {model_config["dec_layers"]}')

    n_query_rows = weights['refpoint_embed.weight'].shape[0]
    if n_query_rows != model_config['num_queries'] * model_config['group_detr']:
        errors.append(f'refpoint_embed has {n_query_rows} rows, but num_queries * group_detr is '
                      f'{model_config["num_queries"] * model_config["group_detr"]}')

    # Architecture fields that were also recorded in the training args
    for k in sorted(args):
        if (k in model_config) and (args[k] != model_config[k]):
            errors.append(f'args has {k}={args[k]!r}, but model_config has {k}={model_config[k]!r}')

    if len(errors) > 0:
        raise ValueError('model_config is inconsistent with the checkpoint:\n' + '\n'.join(errors))

# ...def _check_model_config(...)


#%% Conversion function

def add_model_config(input_file, output_file, resolution=DEFAULT_RESOLUTION, overwrite=False):
    """
    Re-save the checkpoint [input_file] to [output_file], adding model_config metadata.
    See module header for a complete description of what changes.

    Args:
        input_file (str): checkpoint (.pth) file without a model_config
        output_file (str): output .pth file
        resolution (int, optional): the resolution the model was trained at
        overwrite (bool, optional): whether to overwrite [output_file] if it exists

    Returns:
        dict: the checkpoint that was written to [output_file]
    """

    if os.path.exists(output_file) and (not overwrite):
        raise FileExistsError(f'Output file {output_file} exists, use overwrite to replace it')

    print(f'Loading checkpoint: {input_file}')
    checkpoint = torch.load(input_file, map_location='cpu', weights_only=True)

    if 'model_config' in checkpoint:
        raise ValueError(f'{input_file} already contains a model_config')

    model_name = checkpoint['model_name']
    model_cls = getattr(rfdetr, model_name)

    # The classification head has one output per class, plus one for "no object"
    num_classes = checkpoint['model']['class_embed.weight'].shape[0] - 1

    model_config = model_cls._model_config_class(
        resolution=resolution,
        num_classes=num_classes,
    ).model_dump()
    model_config.pop('device', None)
    model_config.pop('pretrain_weights', None)

    _check_model_config(model_config, checkpoint)

    print(f'model_name: {model_name}')
    print(f'resolution: {model_config["resolution"]}')
    print(f'positional_encoding_size: {model_config["positional_encoding_size"]}')
    print(f'num_classes: {num_classes} ({checkpoint["args"]["class_names"]})')

    # Copy every key in its original order, adding model_config right before
    # rfdetr_version, which is where rfdetr's own checkpoint writer puts it
    output = {}
    for k, v in checkpoint.items():
        if k == 'rfdetr_version':
            continue
        if k == 'loops':
            v = dict(v)
            for loop_name in ('validate_loop', 'test_loop'):
                v.setdefault(loop_name, {'state_dict': {}})
        output[k] = v
    output['model_config'] = model_config

    print(f'rfdetr_version: {checkpoint.get("rfdetr_version")} --> {version("rfdetr")}')
    output['rfdetr_version'] = version('rfdetr')

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(output, output_file)
    print(f'Wrote {output_file}')

    return output

# ...def add_model_config(...)


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(
        description='Re-save an RF-DETR checkpoint with model_config metadata'
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Input checkpoint (.pth) without model_config'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Output checkpoint (.pth)'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        default=DEFAULT_RESOLUTION,
        help='Resolution the model was trained at (default: {})'.format(DEFAULT_RESOLUTION)
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the output file if it exists'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    add_model_config(
        input_file=args.input_file,
        output_file=args.output_file,
        resolution=args.resolution,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()
