
from pathlib import Path

from detectron2.config import get_cfg as detectron_get_cfg
from detectron2.model_zoo import model_zoo


def get_cfg(model_weights_path: Path = None, output_path: Path = None, debug: bool = True, num_input_channels: int=1):
    cfg = detectron_get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    if model_weights_path is None:
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = str(model_weights_path)

    cfg.OUTPUT_DIR = str(output_path) if output_path is not None else './output'
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 0 if debug else 6

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 500
    cfg.SOLVER.WARMUP_ITERS = 500 # a warm up is necessary to avoid diverging training while keeping the goal learning rate as high as possible
    cfg.SOLVER.IMS_PER_BATCH = 16 if not debug else 8
    cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    cfg.SOLVER.MAX_ITER = 80000
    cfg.SOLVER.STEPS = (40000, 60000, 70000)
    cfg.SOLVER.GAMMA = 0.5 # after each milestone in SOLVER.STEPS gets reached, the learning rate gets scaled by Gamma.

    cfg.SOLVER.CHECKPOINT_PERIOD = 50 if debug else 3000 #5000
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.TEST.EVAL_PERIOD = 30 if debug else 3000

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.FORMAT = "D" * num_input_channels
    cfg.MODEL.PIXEL_MEAN = [0.5] * num_input_channels
    cfg.MODEL.PIXEL_STD = [1.0] * num_input_channels
    cfg.MIN_AREA = 100

    cfg.DATASETS.TRAIN = ("val",) if debug else ("train",)
    cfg.DATASETS.TEST = ("val",)

    cfg.DEBUG = debug
    return cfg

if __name__ == "__main__":
    cfg = get_cfg()