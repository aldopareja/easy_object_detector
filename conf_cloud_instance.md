The machines are created with ubuntu 16.04


```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

useradd -m aldo
passwd aldo
usermod -aG sudo aldo

sudo fdisk /dev/xvdc
sudo mkfs -t ext4 /dev/xvdc1
sudo echo "/dev/xvdc    /home/aldo/disk1    ext4    defaults     0    2" >> /etc/fstab
sudo mount /dev/xvdc1 /home/aldo/disk1/
sudo chown -R aldo:aldo /home/aldo/disk1/ 

sudo apt-add-repository ppa:fish-shell/release-3
sudo apt-get update
sudo apt-get install fish
usermod -s /usr/bin/fish aldo
su aldo
curl git.io/pure-fish --output /tmp/pure_installer.fish --location --silent
source /tmp/pure_installer.fish; and install_pure


cd ~
mkdir .ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCYwr0QKWkgpxT0mgFg8jTj5SGJFYyy3Acd9+VWH2A73cirCTw42S2+VS3vXJ+mWjuLpSB+0c1Vwh6RGly8UZoI7BNTNZG8LDmGTo5eQhwmpIrcm+F8YCcdPQG8KjGNeRNZ2BMWknErgd/jbvbpvXMZLN15WC0HnkuHkpTO2VnJ6SCyPOLK0g+M7/NfC81H/0oXHPt6nnd3WUIwnytXn8eWsSYyLpZF8/plW5iakRFV7OXMny29yqjOcZvr6pGRE8+X8x06vZAAj6T7fRd+na7aMjRQZoEU65CCd2De756BUgOf95eULY+0eBsnre9nhcYWhrtvgqtGhmfsRTVdoGbP4CdKHv9VS3by9jMjU6G+zsyAaL23sABG/UDo2qXxPnZP9JA4LhgHIuxGncFSbZnFP1V0csBDxordnbGmYRrpl00QBQIufrEttQdP+5aqe2L0hnGJZKjFAU7tThy9x0xV3e2C9O3TcgIT/OsTbUIGI+XpBx2G0lefKMEp0+qQQU4CMCOjpFXffKy6upW4bZhl9xAF776tka+HAqWeEP83t7B1R/G4q3gtSAeYs5fl7OIHrlHLdSuVncnlFR8EBUYFfqj+S4KsrR/oJCoAyEfYXkEXc26SWl3+in/sCjxjXQfoUGGD1k3qoeUPjCMOdi18g9xRNT8Eo+8UYhwVKsha5Q== aldo@aldos-MacBook-Pro.local" >> .ssh/authorized_keys

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
miniconda3/bin/conda init fish

conda env create -f environment.yml
conda activate detector

cd /home/aldo/disk1/
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3.x86_64
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
tar -xzvf MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
chmod a+x MCS-AI2-THOR-Unity-App-v0.3.3.x86_64

wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-passive-physics.zip
sudo apt-get install unzip
unzip training-passive-physics.zip

$ nvidia-xconfig --query-gpu-info

GPU #0:
  Name      : Tesla K80
  UUID      : GPU-d03a3d49-0641-40c9-30f2-5c3e4bdad498
  PCI BusID : PCI:0:23:0
# create the xserver configuration
$ sudo nvidia-xconfig --no-use-display-device --virtual=600x400 --output-xconfig=/etc/X11/xorg.conf --busid=PCI:0:6:0
# launch Xserver
$ sudo /usr/bin/Xorg :0 &
```
wD7UUzf733HM
