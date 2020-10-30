import argparse
from pathlib import Path

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from easy_detector.utils.io import load_cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_out_folder', type=str)
    parser.add_argument('--model_cfg', type=str)
    parser.add_argument('--model_weights', type=str)
    args = parser.parse_args()
    return args

class DetectorPredictor:
    def __init__(self,training_out_path :Path =None,
                 cfg_path:Path=None,
                 model_weigths_path:Path=None,
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

    def __call__(self, input:torch.FloatTensor):
        input_batch =[{'image': input}]
        output = self.model(input_batch)
        assert len(output) == 1
        output = output[0]['instances'].get_fields()
        scores = output['scores']
        boxes = [b.cpu().detach().numpy().astype(np.uint32) for i,b in enumerate(output['pred_boxes'])
                                            if scores[i] >= self.threshold]
        masks = [m.cpu().numpy() for i,m in  enumerate(output['pred_masks'])
                                if scores[i] >= self.threshold]

        return {'boxes': boxes, 'masks': masks}

if __name__ == "__main__":
    args = parse_args()
    predictor = DetectorPredictor(Path(args.training_out_folder))
    test_input = torch.FloatTensor(np.load('example_data/val/inputs/00013044.npy'))
    print(predictor(test_input))