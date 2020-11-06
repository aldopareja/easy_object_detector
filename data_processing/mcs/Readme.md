# Process MCS for detection

## install MCS 0.3.2

Recommended to install in a conda environment or virtualenv

```
pip install git+https://github.com/NextCenturyCorporation/MCS@latest#egg=machine_common_sense
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

