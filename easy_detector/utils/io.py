import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
import hickle as hkl

from detectron2.config import CfgNode


def read_serialized(file_path: Path):
    with open(file_path, "r") as f:
        if file_path.name.endswith(".json"):
            return json.load(f)
        else:
            raise NotImplementedError


def write_serialized(data, file_path: Path):
    """Write json and yaml file"""
    assert file_path is not None
    if file_path.name.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError("Unrecognized filename extension", file_path)


def load_cfg_from_file(cfg_file):
    with open(cfg_file,"r") as f:
        cfg = yaml.load(f)
    cfg = CfgNode(cfg)
    return cfg

def load_mask(mask_path: Path):
    ext = mask_path.suffix
    if ext == '.npy':
        return np.load(mask_path)
    elif ext == '.png':
        return np.array(Image.open(mask_path))
    else:
        raise NotImplementedError

def load_input(input_path: Path):
    if not isinstance(input_path,Path):
        input_path = Path(input_path)
    ext = input_path.suffix
    if ext == '.npy':
        input = np.load(input_path)
    elif ext == '.hkl':
        input = hkl.load(str(input_path))
    else:
        raise NotImplementedError

    return torch.FloatTensor(input)
