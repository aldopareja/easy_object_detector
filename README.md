# easy_object_detector

easy to use object detector. Allows for easy training and inference

# Instalation

For now these are the steps I followed to make it work, please refer to [detectron2 installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) in case of issues:

```
sudo apt-get install ninja-build
pip install torch torchvision opencv-python tensorboard
pip install  'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

# Usage

## Training Data Format

To reduce usage complexity the training requires a specific format. Please refer to [example data](example_data) for an example of the expected input.

Here we used the [ IntPhys ](intphys.com) dataset as an example. [intphys_process.py](data_processing/intphys_process.py) shows an example of the kind of processing needed to enable training.

In general your data folder must have a structure like this:

```
.
+-- _train
    +-- _inputs
    +-- _masks
+-- _val
    +-- _inputs
    +-- _masks    
```

The names of the files in `inputs` anh `masks` should be the same for corresponding input-masks pairs. It should be an integer-castable name (excluding the extension).

`inputs` should contain numpy (.npy) files containing 3d arrays (Channels, Height, Width) of the processed required inputs. In the case of intphys these are single channel arrays where each pixel represents `1/(1+d)` where d is the depth of the pixel in meters.

`masks` should also contain numpy files cotaining 2d uint8 arrays (Height, Width) where the number of unique values in the array correspond to the number of objects in the frame. the set of elements with the same value comprises the segmentation mask of that particular object. The value 0 will be ignored and should be used for background or anything else not required to be detected.

 the validation set should be as close as possible to test conditions and not be more than 7000 images or the evaluation could take too long.
## Training

```
 python -m  easy_detector.train_detector --output_dir ./output  --input_data ./example_data/ --distributed
```

options:
- `--model_weights`: file path pointing to o `.pth` file containing a trained model. If none, a pretrained model on coco_instance_segmentation is used
- `--remove_cache`: recomputes a processed dataset (saved in `./tmp`) and removed any previously computed one
- `--distributed`: uses all available GPUs in the machine to speed up training
- `--num_input_channels`: the number of channels per image/array, default 1

### trainig configuration

the specified training parameters are in [config.py](easy_detector/config.py). The ones included were succesful on Intphys but some tuning might be required if the defaults don't work.

## Inference

After training the model, an instantiation of it can be used to predict new objects in a frame. An example of usage can be found at the end of [predictor.py](easy_detector/predictor.py). In general just instantiate an object `DetectorPredictor` pointing to the output training folder. Calling the object on a 3d tensor with the same format as the training inputs will predict Boxes and Segmentation Masks on the image.

 
