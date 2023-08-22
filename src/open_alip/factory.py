import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import torch

from .constants import DATASET_MEAN, DATASET_STD
from .model import ALIP
from .transform import image_transform


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
        cache_dir: Optional[str] = None,
):
    model_name = model_name.replace('/', '-') 

    if model_name in _MODEL_CONFIGS:
        logging.info(f'Loading {model_name} model config.')
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert False, 'pretrained image towers currently only supported for timm models'

    model = ALIP(**model_cfg)
    pretrained_cfg = {}
   
    model.to(device=device)
    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = pretrained_cfg.get('mean', None) or DATASET_MEAN
    model.visual.image_std = pretrained_cfg.get('std', None) or DATASET_STD

    if jit:
        model = torch.jit.script(model)

    return model


def create_model_and_transforms(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
):
    model = create_model(
        model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        cache_dir=cache_dir)

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(model.visual.image_size, is_train=True, mean=image_mean, std=image_std)
    preprocess_val = image_transform(model.visual.image_size, is_train=False, mean=image_mean, std=image_std)

    return model, preprocess_train, preprocess_val


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
