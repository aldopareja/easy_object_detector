import argparse
import os
import sys
from pathlib import Path

import torch
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager

from easy_detector.config import get_cfg
from easy_detector.detect_dataset import raw_to_detectron
from easy_detector.trainer import CustomDetectTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--rank", type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--remove_cache', action='store_true')
    parser.add_argument('--num_input_channels', type=int)
    return parser.parse_args()

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

def main(args):
    # import ipdb
    # ipdb.set_trace()
    cfg = get_cfg(model_weights_path=Path(args.model_weights),output_path=Path(args.output_dir),
                  debug=args.debug, num_input_channels=args.num_input_channels)
    default_setup(cfg, args)
    raw_to_detectron(Path(args.input_data), args.remove_cache, cfg)
    trainer = CustomDetectTrainer(cfg)
    if args.model_weights is not None:
        trainer.resume_or_load()
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
            machine_rank= args.rank
        )
    else:
        main(args)