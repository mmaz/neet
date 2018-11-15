Create or edit a file in the `.ssh/` directory inside your home directory, called `config`:

`$ nano ~/.ssh/config` or `$ gedit ~/.ssh/config`or `$ vi ~/.ssh/config` (or whichever text editor you are comfortable with)

You will need to add *two* lines to the file, and you will need to replace `PATH_TO_HOME` with the correct path for your platform:

* on linux, `/home/YOURTEAMNAME/.ssh/` (or your username, if you are not using a team laptop)
* on mac, `/Users/YOURUSERNAME/.ssh`
* windows, (TODO)

Add the following two lines:

```
ControlMaster auto
ControlPath PATH_TO_HOME/.ssh/ssh_mux_%h_%p_%r
```

**Important** - remember to start your **FIRST** SSH connection with `-Y` if you plan to use xforwarding (e.g., for `rqt_image_view`)
