'''
example:

python data_processing/intphys_process --input_dir /disk1/intphys_data/dev_meta --output_dir /disk1/detect_data_intphys
'''

import sys

from easy_detector.utils.istarmap_tqdm_patch import array_apply

sys.path.insert(0,'./')
from easy_detector.utils import istarmap_tqdm_patch

import argparse
import os
import shutil
from pathlib import Path
from multiprocessing import cpu_count, Pool
import numpy as np
from PIL import Image

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    return args


def build_recursive_case_paths(input_folder, cases):
    if "scene" not in os.listdir(input_folder):
        to_recurse = sorted(list(os.path.join(input_folder, sub_folder) for sub_folder in os.listdir(input_folder)))
        for new_folder in to_recurse:
            if os.path.isdir(new_folder):
                build_recursive_case_paths(new_folder, cases)
    else:
        cases.append(Path(input_folder))
    return cases


# def create_dirs(vid_num: str, out_dir: str):
#     vid_path = Path(out_dir) / vid_num.zfill(6)
#     shutil.rmtree(vid_path, ignore_errors=True)
#     os.makedirs(vid_path)
#
#     input_path = vid_path / 'input'
#     os.makedirs(input_path)
#
#     segmentation_path = vid_path / 'segmentation'
#     os.makedirs(segmentation_path)
#
#     return {'vid': vid_path,
#             'input': input_path,
#             'segment': segmentation_path}


def process_frame(depth_file, mask_file, out_dir, index):
    depth_array = np.asarray(Image.open(depth_file), dtype=np.float)
    depth_array = (2 ** 16 - 1 - depth_array) / 1000.0
    depth_array = np.expand_dims(1 / (1 + depth_array), axis=0)
    # Intphys  encoding  here: https://www.intphys.com/benchmark/training_set.html
    np.save(out_dir / 'inputs' / ( str(index).zfill(8) + '.npy' ), depth_array)

    mask = np.asarray(Image.open(mask_file))
    np.save(out_dir / 'masks' / ( str(index).zfill(8) + '.npy' ), mask)


def process_video(video_path, vid_num, out_dir):
    depths = []
    masks = []
    for i in range(1, 101):
        depths.append(video_path / 'depth' / ('depth_' + str(i).zfill(3) + '.png'))
        masks.append(video_path / 'masks' / ( 'masks_' + str(i).zfill(3) + '.png' ))

    [process_frame(d, m, out_dir, vid_num * 1000 + f_num) for f_num, (d, m) in enumerate(zip(depths, masks))]


if __name__ == '__main__':
    args = parse_args()
    video_folders = build_recursive_case_paths(args.input_dir, [])
    shutil.rmtree(args.output_dir, ignore_errors=True)
    [os.makedirs(args.output_dir / a) for a in ['inputs', 'masks']]
    worker_args = [(v, i, args.output_dir) for i, v in enumerate(video_folders)]
    array_apply(process_video, worker_args,args.parallel)
        # with Pool(int(cpu_count())) as p:
        #     # for _ in tqdm(p.istarmap(process_video, worker_args),
        #     #               total=len(worker_args)):
        #     #     pass
        #     [_ for _ in tqdm(p.istarmap(process_video,worker_args),
        #                      total=len(worker_args))]
        #     # p.starmap(process_video, worker_args)
