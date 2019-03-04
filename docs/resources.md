# Resources


!!! warning "You can contribute to this list!"
    *Do you have tips, resources, or corrections to add? Did you find something frustrating or unhelpful and want to remove it? Notice any typos?* 

    Pull-requests welcome! Submit one at <https://github.com/mmaz/neet>

UAV Course website:

<http://neet.beaver.works>

If this is your first time using a shell, git, or ssh, here are a few tutorials to help acquaint yourself with the basics.

**Tip:** *2x speed videos* On youtube, click the gear icon and change the playback speed to 2x, which is useful when watching slower-paced videos.

### Python
MIT's own [intro to python and programming](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-videos/) course is a good place to review python basics, such as branching, iteration, and lists.

Videos are also here:[youtube playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP63WbdFxL8giv4yhgdMGaZNA)


### The Shell

The shell (aka, command line or terminal) is how we connect to the UAV.
 
Shell basics:

* <https://www.youtube.com/watch?v=poT5Yd0Ag8I>
* <https://www.youtube.com/watch?v=oxuRxtrO2Ag>
* <https://www.digitalocean.com/community/tutorials/how-to-use-cd-pwd-and-ls-to-explore-the-file-system-on-a-linux-server>
 
Some useful commands to learn are:

`ls, cd, pwd, find, grep, mkdir, rm, mv, cp, less, cat, which`
 
Shell operators like `|` (the pipe symbol) and `>` as well as control signals, like `Ctrl+C`, are also useful to know.
 
You may need to occasionally find your IP address and check for network reachability. Some useful commands for this:

`ping, ifconfig, wget, curl`

For example,

```shell
$ ping drone.beaver.works
```
 
I often find my IP address with the following command:
 
`ifconfig | grep inet`
 
How does this work? `ifconfig` spits out a lot of network configuration information. The `|` symbol pipes the output of `ifconfig` to `grep`, which searches for the word 'inet' on each line in the output. This corresponds to a list of IP addresses associated with my computer. On linux you can also use `hostname -I` – in general there can be many ways to find the same information using the command line, with different tradeoffs.
 
This cheatsheet (or others, just google for 'bash cheat sheet') may come in handy:
 
<https://gist.github.com/LeCoupa/122b12050f5fb267e75f>
 
### SSH and SCP:

<https://www.youtube.com/watch?v=rm6pewTcSro>
 
### Git

* <https://www.youtube.com/watch?v=zbKdDsNNOhg> (answers to “why use Git in the first place?”)
* <https://www.youtube.com/watch?v=3a2x1iJFJWc>
* <https://youtu.be/9pa_PV2LUlw>
 
### Writing Python code 
 
You’ll want to pick a code editor. Visual Studio Code, Sublime, Atom, GEdit, vim, and emacs are all popular choices.

<https://code.visualstudio.com/> is easy to install and configure, available on all platforms, and free. 

 
### ROS (Robot Operating System)

One-hour introduction to ROS:

<https://www.youtube.com/watch?v=0BxVPCInS3M>
