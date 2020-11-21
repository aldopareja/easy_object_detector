from pathlib import Path

from detectron2.structures import BoxMode

from easy_detector.utils.io import read_serialized
from easy_detector.utils.istarmap_tqdm_patch import array_apply


def fix_from_serialization(data_dict):
    data_dict['file_name'] = data_dict['file_name']
    for an in data_dict['annotations']:
        an['bbox_mode'] = BoxMode.XYXY_ABS
        an['segmentation']['counts'] = an['segmentation']['counts'].encode('ascii')
    return data_dict


def get_data_dicts(coco_path):
    data_dicts = read_serialized(coco_path)
    data_dicts = array_apply(fix_from_serialization, data_dicts, use_tqdm=False, parallel=True, unpack=False)
    return data_dicts
