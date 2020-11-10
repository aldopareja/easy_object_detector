from pathlib import Path

import numpy as np
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from easy_detector.utils.io import load_input


def visualize_data_dict(d, save_path: Path=None, channels=(0,1,2), swaps=((0,1), (1,2))):
    # img = (load_input(d['file_name']) * 255)[:3].numpy()
    # img = img.swapaxes(0, 1).swapaxes(1, 2)
    # assumes d['file_name'] is in (C, H, W) format as required by detectron2
    # #the visualizer requires everything in rgb and (H, W, C) version
    img = (load_input(d['file_name']) * 255)[list(channels)].numpy()
    if img.shape[0] == 1:
        img = np.concatenate([img]*3, axis=0)
    assert img.shape[0] == 3
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = img.astype(np.uint8)
    visualizer = Visualizer(img, metadata=MetadataCatalog.get('val'))
    out = visualizer.draw_dataset_dict(d).get_image()
    if save_path:
        Image.fromarray(out).save(save_path)
    return out

