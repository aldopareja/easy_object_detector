# Process MCS for detection

## requirements

```
python 3.8.5
cuda 10.2 installed (not the one that comes with pytorch)
```

for cuda follow [ these docs ](https://developer.nvidia.com/cuda-10.2-download-archive) depending on your system. Only `deb (local)` might work since the network versions install the latest cuda release. 

## Detectron2 for debugging and visualization

```
pip install torch==1.6.0 torchvision==0.7.0
pip install git+https://github.com/facebookresearch/detectron2.git@v0.2.1
```

## install MCS 0.3.3

Recommended to install in a conda environment or virtualenv

```
pip install git+https://github.com/NextCenturyCorporation/MCS@0.3.3
```

## other dependencies

```
pip install h5py==2.10.0 hickle==4.0.1 opencv-python==4.4.0.44 ipdb pillow numpy
```

the version of `h5py` is particularly important or it'll break `hickle`

## download data and everything else

```
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3.x86_64
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
tar -xzvf MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
chmod a+x MCS-AI2-THOR-Unity-App-v0.3.3.x86_64

wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-passive-physics.zip
sudo apt-get install unzip
unzip training-passive-physics.zip

```

## remote rendering

First set up an Xserver:

1. query the GPU info and take note of the PCI-bus
```
$ nvidia-xconfig --query-gpu-info

GPU #0:
  Name      : Tesla V100-PCIE-16GB
  UUID      : GPU-e138b3eb-3c23-777f-9fcf-fa808647ea95
  PCI BusID : PCI:175:0:0

  Number of Display Devices: 0
```

2. rewrite the Xserver configuration

```
$ sudo nvidia-xconfig --no-use-display-device --virtual=600x400 --output-xconfig=/etc/X11/xorg.conf --busid=PCI:175:0:0

Using X configuration file: "/etc/X11/xorg.conf".
Backed up file '/etc/X11/xorg.conf' as '/etc/X11/xorg.conf.backup'
New X configuration file written to '/etc/X11/xorg.conf'
```

3. start an Xserver on display number :0 

```
sudo /usr/bin/Xorg :0 &
```

4. Set DEBUG in mcs_process to TRUE `data_processing/mcs/mcs_process.py:31`

5. Run the processor in a debug session. Point each argument to yours depending on your folder structure.

from the repository root:

```
python -m ipdb data_processing/mcs/mcs_process.py --mcs_executable /home/aldopareja/MCS/MCS-AI2-THOR-Unity-App-v0.3.2.x86_64 --data_path /home/aldopareja/MCS/evaluation3Training/ --output_dir /disk1/mcs_physics_data --num_parallel_controllers 0
```

6. the images in `tmp_mcs_images` will have the mentioned problem 
