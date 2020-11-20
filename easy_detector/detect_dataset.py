import argparse
import os
import re
import shutil
from itertools import repeat
from pathlib import Path

import numpy as np
import pycocotools
from detectron2.utils import comm
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from easy_detector.utils.data_utils import get_data_dicts
from easy_detector.utils.io import write_serialized, load_mask, load_input
from easy_detector.utils.istarmap_tqdm_patch import array_apply
from easy_detector.utils.visualize import visualize_data_dict

DEBUG = False

def process_frame(input_file: Path, masks_file: Path, min_area):
    frame = {}
    masks = load_mask(masks_file)

    frame['file_name'] = str(input_file)
    frame['image_id'] = int(re.sub('[^0-9]', '', input_file.name.split('.')[0]))
    frame['height'], frame['width'] = map(int, masks.shape)

    objs = []
    for m_id in np.unique(masks):
        if m_id == 0:
            continue
        obj_mask = masks == m_id
        if obj_mask.sum() > min_area:
            mask_y, mask_x = obj_mask.nonzero()
            bbox = list(map(int, [mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()]))
            if bbox[3] <= bbox[1] + 2 and bbox[2] <= bbox[0] + 2:  # width and height shouldn't be too small
                continue
            encoded_mask = pycocotools.mask.encode(np.asarray(obj_mask, order="F"))
            encoded_mask["counts"] = encoded_mask["counts"].decode('ascii')
            objs.append({"bbox": bbox,
                         "bbox_mode": int(BoxMode.XYXY_ABS),
                         "segmentation": encoded_mask,
                         "category_id": 0,
                         "iscrowd": 0})

    frame["annotations"] = objs
    if DEBUG:
        visualize_data_dict(frame, save_path=Path('erase.png'), channels=(3,)) #TODO: remove
    return frame


def raw_to_detectron(data_path: Path, remove_cache: bool, cfg: CfgNode):
    data_splits = ['val']
    data_splits += ['train'] if not cfg.DEBUG else []
    for name in data_splits:
        coco_path = Path('.') / 'tmp' / ('coco_' + name + '.json')

        if (remove_cache or not coco_path.exists()) and comm.is_main_process():
            input_files = [a for a in (data_path / name / 'inputs').iterdir()]
            mask_ext = next((data_path / name / 'masks').iterdir()).name.split('.')[1]
            mask_files = [a.parent.parent / 'masks' / (a.name.split('.')[0] + '.' + mask_ext)
                          for a in input_files]
            shutil.rmtree(coco_path, ignore_errors=True)
            coco_path.parent.mkdir(parents=True, exist_ok=True)
            frame_objects = array_apply(process_frame, zip(input_files, mask_files, repeat(cfg.MIN_AREA)),
                                        parallel=not DEBUG,
                                        total=len(input_files),
                                        chunksize=1000)
            write_serialized(frame_objects, coco_path)

        DatasetCatalog.register(name, lambda d=coco_path: get_data_dicts(d))
        MetadataCatalog.get(name).set(thing_classes=['object'])
    comm.synchronize()
    # TODO: do we need to set json_file as well?, might not be necessary


if __name__ == "__main__":
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.MIN_AREA = 25
    cfg.DEBUG = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    data_path = Path(parser.parse_args().data_path)
    raw_to_detectron(data_path, True, cfg)
