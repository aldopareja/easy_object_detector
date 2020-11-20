import copy
import os
from pathlib import Path

import numpy as np
import torch
from detectron2.data import build_detection_train_loader, build_detection_test_loader, detection_utils
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from easy_detector.utils.io import load_input
from easy_detector.utils.visualize import visualize_data_dict

DEBUG = False


class DetectionMapper():
    def __call__(self, dataset_dict):
        if DEBUG:
            visualize_data_dict(dataset_dict, save_path=Path('erase.png'))

        dataset_dict = copy.deepcopy(dataset_dict)
        dataset_dict['image'] = load_input(dataset_dict['file_name'])
        image_shape = (dataset_dict["height"], dataset_dict["width"])

        annos = dataset_dict["annotations"]
        instances = detection_utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        dataset_dict["instances"] = instances
        return dataset_dict


class CustomDetectTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, DetectionMapper())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, DetectionMapper())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
