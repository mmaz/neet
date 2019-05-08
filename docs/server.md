# NEET Spring 2019 Server

!!! danger "Keep this in mind!"
     The NEET Server is a shared resource! Coordinate the times you are using the server with the other NEET teams so that you do not overlap. And **remember to shut down your Jupyter Notebook or training script** after you are done. Do not leave a training script running - it will reserve all available GPU memory, and prevent other teams from training.

     If you and another team agree beforehand, you can limit the memory TensorFlow allocates while training, and both teams can train on the server simultaneously. Still, in this case, remember to shut down jupyter or your python training script when you are done.

     Ask the instructors before running any commands with `sudo` rights.

     Lastly, please do **not** reboot the server! Again, ask if you have questions.

## Installing Dependencies

There's nothing to do! The dependencies you need for the **Imitation learning lab** and the **Reinforcement learning lab** are already installed in the `base` conda environment. This environment is automatically activated when you log in, so there are no extra steps necessary (i.e., you do not need to run `conda activate base`).

(`conda list` will show all dependencies installed in the environment if you are curious.)

## Connecting to the server over SSH

At the terminal, the following command will SSH into the server, enable x-forwarding, and also forward a port for Jupyter Notebook access:

!!! note
    You will need to replace `$USERNAME` and `$IP_ADDRESS` below with the appropriate values (ask the instructors in the Slack chatroom)

```shell
ssh -L 8888:localhost:8888 -X  $USERNAME@$IP_ADDRESS
```

Once you are connected, run `ls` - you have an empty directory for your team already created in the home directory. (`mx0` is the "instructor" directory)

```shell
$ ls
donkey_simulator mx0  mx1  mx2  mx3 
```

### Adding a shortcut for easy SSH access:

**This step is optional**

You can simplify the process of connecting to the NEET server further by adding the following information to your **local** SSH config file:

!!! danger "Warning"
    Make sure you are editing the config file __on your personal laptop__ (not the server!) before proceeding:

    `$ vi ~/.ssh/config`

!!! note
    You will need to replace `$USERNAME` and `$IP_ADDRESS` below with the appropriate values (ask the instructors in the Slack chatroom)

```
Host neet
    User $USERNAME
    HostName $IP_ADDRESS
    ForwardX11 yes
    IdentityFile ~/.ssh/id_rsa
    LocalForward 8888 127.0.0.1:8888
    ControlPath ~/.ssh/controlmasters_%r@%h:%p
    ControlMaster auto
```

This sets up port-forwarding (for Jupyter), [SSH multiplexing](https://en.wikibooks.org/wiki/OpenSSH/Cookbook/Multiplexing#Setting_Up_Multiplexing), and X-forwarding once and for all.

Afterwards you can SSH into the server just by typing:

```shell
$ ssh neet
```

at your local command prompt.

### Convenient SCP

With the shortcut, you can also SCP directories and files to the server easily, e.g., to copy to your team's directory `mxN` (where `N` is 1,2, or 3):

```shell
$ scp -r local_directory_w_training_images neet:~/mxN/
```

### No-password logins

If you would like to avoid being asked for a password each time, you can generate a local identity file:

<https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>

Then, use the following command to connect to the server.

```shell
# only required once
$ ssh-copy-id neet
```

which will copy your public key to the server.

## Opening Jupyter Notebooks over SSH

!!! note
     If you have Jupyter Notebook running locally, shut it down first (or restart it and change the port to something besides the default port `8888`, so that the local port used by Jupyter does not conflict with the SSH-forwarded one)

     **On your local computer:**

     `jupyter notebook --port 8889`
    
     for example.

Once you have SSHed into the computer, you can start jupyter in your team's folder (include `--no-browser` so that Firefox does not try to load over SSH - which will be annoyingly slow)

```shell
$ cd mxN/
$ jupyter notebook --no-browser
```

Jupyter will provide a URL for you to use in your local computer's browser. Copy and paste it into your browser, e.g.,

```
Or copy and paste one of these URLs:
   http://localhost:8888/?token=r4nd0mh3xstring
```

!!! note
    It is highly recommended to run a program like `tmux` or `screen` after first logging in, so that your work is not lost in case your SSH connection is interrupted. `tmux` is already installed on the server. 

    * [Here](https://danielmiessler.com/study/tmux/) is a simple guide that introduces `tmux`. 
    * Also, [here](https://gist.github.com/andreyvit/2921703) is a cheatsheet for quick reference.

```shell
$ cd mxN/
$ tmux
$ jupyter notebook --no-browser
```

## Running the RL Simulation Environment

You can collect data and train a policy for the RL lab in person on the server (contact the instructors over Slack regarding server access).

There is a folder already created in the home directory called `~/donkey_simulator/` - you can start the simulator by double-clicking the executable `build_sdsandbox.x86_64` in the Ubuntu file explorer or by running it in a new terminal.

Once the simulator is running, you can train a policy in another terminal or over SSH.

!!! danger "Remember"
     Shut down the simulator and your training scripts when you are done!


