# MIT NEET Fall 2018

## Multiplex SSH

This is highly recommended for connecting to the drones:

Create or edit a file in the `.ssh/` directory inside your home directory, called `config`:

`$ nano ~/.ssh/config` or `$ gedit ~/.ssh/config`or `$ vi ~/.ssh/config` (or whichever text editor you are comfortable with)

You will need to add *two* lines to the file, and you will need to replace `PATH_TO_HOME` with the correct path for your platform:

* on linux, `/home/YOURTEAMNAME/.ssh/` (or your username, if you are not using a team laptop)
* on mac, `/Users/YOURUSERNAME/.ssh`
* windows, (TODO)

Add the following two lines (**Remember:** change `PATH_TO_HOME` as specified above):

```
ControlMaster auto
ControlPath PATH_TO_HOME/.ssh/ssh_mux_%h_%p_%r
```

**Important** - remember to start your **FIRST** SSH connection with `-Y` if you plan to use xforwarding (e.g., for `rqt_image_view`)


## Directory Setup

 These instructions replace [this section](https://bwsi-uav.github.io/website/student_drone_setup.html#Directory-Setup) on the website.

On the SSH window on your team laptop, enter the following commands

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

## Compressed Camera Feeds

To find if your drone supports compressed camera feeds:

0. Start `roscore`
1. Start optical flow: `sudo -E ~/bwsi-uav/catkin-ws/src/aero-optical-flow/build/aero-optical-flow`
2. `$ rostopic list | grep compressed`

If you don't see `/aero_downward_camera/image/compressed` in the results you will need to install compressed transport support:

`sudo apt-get install ros-kinetic-image-transport-plugins`

then **restart your camera feed** by restarting the `aero-optical-flow` binary (step 1 above).

To record a compressed downward camera feed:

```bash
$ cd ~/rosbags/ # or wherever you want to store your rosbag
$ time rosbag record -O downward /aero_downward_camera/image/compressed # -O specifies the filename
```

You can then SCP your rosbag to your team laptop.

To convert compressed camera messages to OpenCV images, you can't use CVBridge. Here is an OpenCV-specific decoding solution. (You could also use CompressedImage from `sensor_msgs.msg`):

```python
from __future__ import print_function
import cv2
import numpy as np
import roslib
import rospy

from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge since it does not support CompressedImage
# from cv_bridge import CvBridge, CvBridgeError

import rosbag
import os

DEST = "/path/to/folder/to/save/images"
BAG  = "/path/to/rosbag.bag"
#your camera topic:
CAM = '/aero_downward_camera/image/compressed'

def bag2msgs():
    bag = rosbag.Bag(BAG)
    if bag is None:
        raise ValueError("no bag {}".format(BAG))
    msgs = []
    for topic, msg, t in bag.read_messages(topics=[CAM]):
        msgs.append(msg)
    bag.close()
    print("MESSAGES: {}".format(len(msgs)))
    return msgs

def uncompress(msgs):
    imgs = []
    for msg in msgs:
         #### direct conversion to CV2 ####
         np_arr = np.fromstring(msg.data, np.uint8)
         image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
         imgs.append(image_np)
    return imgs


if __name__ == "__main__":

    if os.listdir(DEST) != []:
        raise ValueError('need empty directory for dest {}'.format(DEST))
    
    msgs = bag2msgs()
    imgs = uncompress(msgs)

    for idx,im in enumerate(imgs):
        if idx % 50 == 0:
            print(idx)
        imname = "frame{:05d}.jpg".format(idx)
        cv2.imwrite(DEST + imname, im)
```