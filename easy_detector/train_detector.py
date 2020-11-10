import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from detectron2.engine import launch

from easy_detector.config import get_cfg
from easy_detector.detect_dataset import raw_to_detectron
from easy_detector.trainer import CustomDetectTrainer
from easy_detector.utils.detectron_setup import default_setup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    # parser.add_argument("--rank", type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--remove_cache', action='store_true')
    parser.add_argument('--num_input_channels', type=int)
    return parser.parse_args()


def main(args):
    if args.debug and not args.distributed:
        import ipdb
        ipdb.set_trace()

    cfg = get_cfg(model_weights_path=Path(args.model_weights) if args.model_weights is not None else None,
                  output_path=Path(args.output_dir),
                  debug=args.debug,
                  num_input_channels=args.num_input_channels)
    default_setup(cfg, args)
    raw_to_detectron(Path(args.input_data), args.remove_cache, cfg)
    trainer = CustomDetectTrainer(cfg)
    if not args.resume:
        shutil.rmtree(cfg.OUTPUT_DIR)
    # resume = args.model_weights is not None
    trainer.resume_or_load(args.resume)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if args.distributed and num_gpus>1:
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        launch(
            main,
            num_gpus,
            args= (args,),
            num_machines=1,
            dist_url= "tcp://127.0.0.1:{}".format(port),
            machine_rank=0
        )
    else:
        main(args)