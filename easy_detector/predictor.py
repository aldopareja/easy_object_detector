import argparse
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from easy_detector.detect_dataset import process_frame
from easy_detector.utils.io import load_cfg_from_file, load_input
from easy_detector.utils.visualize import visualize_data_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_out_folder', type=str)
    # parser.add_argument('--model_cfg', type=str)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--num_examples', type=int)
    args = parser.parse_args()
    return args


class DetectorPredictor:
    def __init__(self, training_out_path: Path = None,
                 cfg_path: Path = None,
                 model_weigths_path: Path = None,
                 threshold=0.9):
        if training_out_path is not None:
            cfg_path = training_out_path / 'config.yaml' if cfg_path is None else cfg_path
            model_weigths_path = training_out_path / 'model_final.pth' if model_weigths_path is None \
                else model_weigths_path
        cfg = load_cfg_from_file(cfg_path)
        cfg.MODEL.WEIGHTS = str(model_weigths_path)

        self.cfg = cfg
        self.model = build_model(self.cfg).eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.threshold = threshold

    def __call__(self, input: torch.FloatTensor):
        input_batch = [{'image': input}]
        output = self.model(input_batch)
        assert len(output) == 1
        output = output[0]['instances'].get_fields()
        scores = output['scores']
        boxes = [b.cpu().detach().numpy().astype(np.uint32) for i, b in enumerate(output['pred_boxes'])
                 if scores[i] >= self.threshold]
        masks = [m.cpu().numpy() for i, m in enumerate(output['pred_masks'])
                 if scores[i] >= self.threshold]

        return {'boxes': boxes, 'masks': masks}


def predict_data_dict(detector: DetectorPredictor, input_tensor):
    frame = {}
    frame['file_name'] = ''
    frame['image_id'] = 0
    _, frame['height'], frame['width'] = input_tensor.shape

    # input_tensor = load_input(input_file)
    out = detector(input_tensor)
    anns = []
    for m, b in zip(out['masks'], out['boxes']):
        encoded_mask = mask_util.encode(np.asfortranarray(m))
        anns.append({"bbox": b,
                     "bbox_mode": BoxMode.XYXY_ABS,
                     "segmentation": encoded_mask,
                     "category_id": 0,
                     "iscrowd": 0})
    frame['annotations'] = anns
    return frame


if __name__ == "__main__":
    args = parse_args()
    predictor = DetectorPredictor(Path(args.training_out_folder))

    input_files = [a for a in (Path(args.input_data) / 'val' / 'inputs').iterdir()]

    shutil.rmtree(Path('inference_examples'), ignore_errors=True)
    Path('inference_examples').mkdir()

    for idx, in_file in enumerate(random.sample(input_files, args.num_examples)):
        input_tensor = load_input(in_file)
        frame_dict = predict_data_dict(predictor, input_tensor)
        visualize_data_dict(frame_dict,
                            save_path=Path('inference_examples') / (str(idx) + '.png'),
                            input_tensor=input_tensor)
