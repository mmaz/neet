To find if your drone supports compressed camera feeds:

0. Start `roscore`
1. Start `aero-optical-flow`
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
