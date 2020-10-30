import copy
import os

import numpy as np
import torch
from detectron2.data import build_detection_train_loader, build_detection_test_loader, detection_utils
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class DetectionMapper():
    # def __init__(self, cfg):
    #     self.use_dept = cfg.USE_DEPTH
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        dataset_dict['image'] = torch.FloatTensor(np.load(dataset_dict['file_name']))
        # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # img = read_image(dataset_dict["file_name"])
        # if self.use_dept:
        #     img = torch.FloatTensor(1 / (1 + img))
        #     dataset_dict["image"] = img.unsqueeze(dim=0)
        # else:
        #     img = torch.FloatTensor(img)/255.0
        #     dataset_dict["image"] = img
        # img = torch.FloatTensor(1 / (1 + img)) if self.use_dept else torch.FloatTensor(img)
        # dataset_dict["image"] = img.unsqueeze(dim=0)
        image_shape = (dataset_dict["height"], dataset_dict["width"])

        annos = dataset_dict["annotations"]
        # annos = filter_non_coco_annotations(annos)
        instances = detection_utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        dataset_dict["instances"] = instances
        return dataset_dict

class CustomDetectTrainer(DefaultTrainer):
    # def __init__(self,cfg):
    #     super().__init__(cfg)
    #     if cfg.RESUME:
    #         self.resume_or_load(resume=True)

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
