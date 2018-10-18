To find if your drone supports compressed camera feeds:

0. Start `roscore`
1. Start optical flow: `sudo -E ~/bwsi-uav/catkin-ws/src/aero-optical-flow/build/aero-optical-flow`
2. `$ rostopic list | grep compressed`

If you don't see `/aero_downward_camera/image/compressed` in the results you will need to install compressed transport support:

`sudo apt-get install ros-kinetic-image-transport-plugins`

then **restart your camera feed** by restarting the `aero-optical-flow` binary.

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
