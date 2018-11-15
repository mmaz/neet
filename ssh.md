Create or edit a file in the `.ssh` directory called `config`, i.e.,

`$ nano ~/.ssh/config`

and add the following lines, replacing `PATH_TO_HOME` with the correct path for your platform:

* on linux, `/home/YOURTEAMNAME/.ssh/`
* on mac, `/Users/YOURUSERNAME/.ssh`
* windows, (TODO)

```
ControlMaster auto
ControlPath PATH_TO_HOME/.ssh/ssh_mux_%h_%p_%r
```

**Important** - remember to start your **FIRST** SSH connection with `-Y` if you plan to use xforwarding (e.g., for `rqt_image_view`)
