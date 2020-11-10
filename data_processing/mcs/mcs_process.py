"""
this files uses MCS 0.3.2 and passive intphys scenes and dumps data to train a detector
"""
import os
import shutil
import sys
import random

from machine_common_sense import Controller

from easy_detector.detect_dataset import process_frame
from easy_detector.utils.visualize import visualize_data_dict

sys.path.insert(0, './')
from easy_detector.utils.istarmap_tqdm_patch import array_apply

import argparse
from pathlib import Path
from time import sleep
from itertools import repeat

import h5py
import machine_common_sense as mcs
import numpy as np
from PIL import Image
import hickle as hkl

from multiprocessing import Process, Queue, cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcs_executable', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_parallel_controllers', type=int)
    args = parser.parse_args()
    args.data_path = Path(args.data_path)
    args.input_dir = Path(args.mcs_executable)
    args.output_dir = Path(args.output_dir)
    return args

DEBUG = True

def dump_for_detectron(step_data, out_path, index):
    # print(step_data)
    depth: np.ndarray = step_data.depth_map_list[0]
    depth = 1 / (1 + depth)
    rgb = np.array(step_data.image_list[0], dtype=np.float32) / 255.0
    input = np.concatenate([rgb, depth[..., np.newaxis]], axis=2)
    input = input.swapaxes(2, 0).swapaxes(1, 2) # now it is C, H, W

    masks = np.array(step_data.object_mask_list[0])
    masks = masks[:, :, 0] + masks[:, :, 1] * 256 + masks[:, :, 2] * 256 ** 2
    assert not (masks == 0).any(), "there shouldn't be 0s or they will be ignored"

    out_mask = np.zeros(masks.shape, dtype=np.uint8)
    assert len(out_mask.shape) == 2

    foreground_ids = [e.color['r'] + e.color['g'] * 256 + e.color['b'] * 256 ** 2
                      for e in  step_data.structural_object_list
                      if not (e.uuid.startswith('wall')  or e.uuid.startswith('floor'))]

    foreground_ids += [e.color['r'] + e.color['g'] * 256 + e.color['b'] * 256 ** 2
                      for e in  step_data.object_list]

    id = 1
    for v in foreground_ids:
        indices = masks == v
        out_mask[indices] = id
        id += 1
    # assert not (out_mask == 0).any()

    input_to_file = out_path / 'inputs' / (str(index).zfill(9) + '.hkl')
    # input_to_file = out_path / 'inputs' / (str(index).zfill(9) + '.npy')

    hkl.dump(input, input_to_file, mode='w', compression='gzip')
    # np.save(input_to_file, input)

    mask_file = out_path / 'masks' / (str(index).zfill(9) + '.png')
    Image.fromarray(out_mask).save(str(mask_file))

    if DEBUG:
        frame_dict = process_frame(input_to_file, mask_file, 100)
        visualize_data_dict(frame_dict, Path('erase.png'))


def process_scene(controller, scene_path, output_path, vid_index, concurrent, tp: ThreadPoolExecutor):
    config_data, _ = mcs.load_config_json_file(scene_path)

    jobs = []
    frame_id = 0
    step_data = controller.start_scene(config_data)
    if concurrent:
        jobs.append(tp.submit(dump_for_detectron, step_data,
                              output_path,
                              vid_index * 500 + frame_id))
    else:
        jobs.append(dump_for_detectron(step_data, output_path, vid_index * 500 + frame_id))

    frame_id += 1
    actions = config_data['goal']['action_list']
    for a in actions:
        assert len(a) == 1, "there must be an action"
        step_data = controller.step(a[0])
        if concurrent:
            jobs.append(tp.submit(dump_for_detectron, step_data,
                                  output_path,
                                  vid_index * 500 + frame_id))
        else:
            jobs.append(dump_for_detectron(step_data, output_path, vid_index * 500 + frame_id))

        frame_id += 1

    controller.end_scene("classification", 0.0)
    if concurrent:
        [j.result() for j in jobs]


class SequentialSceneProcessor:
    def __init__(self, mcs_executable: Path, concurrent_dump: bool):
        self.controller = mcs.create_controller(str(mcs_executable),
                                                depth_maps=True,
                                                object_masks=True,
                                                history_enabled=False)

        self.concurrent = concurrent_dump
        self.tp = ThreadPoolExecutor(4)

    def process(self, w_arg):
        (s, _, o, v) = w_arg
        process_scene(self.controller, s, o, v, self.concurrent, self.tp)


def ParallelSceneProcess(q: Queue, mcs_executable: Path, concurrent_dump):
    controller = mcs.create_controller(str(mcs_executable),
                                       depth_maps=True,
                                       object_masks=True,
                                       history_enabled=False)
    with ThreadPoolExecutor(4) as p:
        while True:
            w_arg = q.get()
            if w_arg is None:
                break
            (s, _, o, v) = w_arg
            process_scene(controller, s, o, v, concurrent_dump, p)


if __name__ == "__main__":
    args = parse_args()
    scene_files = [args.data_path / a for a in args.data_path.iterdir()]

    shutil.rmtree(args.output_dir, ignore_errors=True)
    training_path = args.output_dir / 'train'
    (training_path / 'inputs').mkdir(parents=True, exist_ok=True)
    (training_path / 'masks').mkdir(parents=True, exist_ok=True)

    w_args = [(s, e, o, i) for i, (s, e, o) in enumerate(zip(scene_files,
                                                             repeat(args.mcs_executable),
                                                             repeat(training_path)))]

    if args.num_parallel_controllers > 0:
        work_queue = Queue()

        workers = [Process(target=ParallelSceneProcess,
                           args=(work_queue, args.mcs_executable, True)) for _ in range(args.num_parallel_controllers)]
        [w.start() for w in workers]

        w_args = [work_queue.put(w) for w in w_args]

        [work_queue.put(None) for _ in range(args.num_parallel_controllers)]
        [w.join() for w in workers]
        work_queue.close()
    else:
        worker = SequentialSceneProcessor(args.mcs_executable, False)
        [worker.process(w_arg) for w_arg in w_args]

    [(args.output_dir / 'val' / d).mkdir(parents=True, exist_ok=True)
     for d in ('inputs', 'masks')]

    val_input_paths = random.sample([*(training_path / 'inputs').iterdir()], 5000)
    for input_path in val_input_paths:
        dest_input_path = args.output_dir / 'val' / 'inputs' / input_path.name
        shutil.move(input_path, dest_input_path)

        mask_path = training_path / 'masks' / (input_path.name.split('.')[0] + '.png')
        dest_mask_path = args.output_dir / 'val' / 'masks' / mask_path.name
        shutil.move(mask_path, dest_mask_path)

    #kill stalling controllers
    os.system('pkill -f MCS-AI2-THOR-Unity-App')
