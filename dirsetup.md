
## Directory Setup

1. On the SSH window on your team laptop, enter the following commands


```bash
cd ~
cd ~/bwsi-uav/catkin_ws/src
git clone https://github.com/BWSI-UAV/aero_control.git
cd aero_control
git remote add upstream https://github.com/BWSI-UAV/aero_control.git 
cd ~/bwsi-uav
git clone https://github.com/BWSI-UAV/laboratory.git
cd laboratory
git remote add upstream https://github.com/BWSI-UAV/laboratory.git 
cd ~/bwsi-uav
git clone https://github.com/BWSI-UAV/documents.git
cd documents
git remote add upstream https://github.com/BWSI-UAV/documents.git 
```
