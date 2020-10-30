import json
from pathlib import Path

import yaml

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