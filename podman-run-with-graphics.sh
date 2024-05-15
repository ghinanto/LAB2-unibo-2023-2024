#!/bin/sh
podman run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm -it -v $HOME:/userhome --user $(id -u) $*
