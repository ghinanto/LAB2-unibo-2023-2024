#!/bin/sh
# Add container name at the end and eventually command to run in container at the very end.
# ex. podman-run-with-graphics.sh rootproject/root
# ex. podman-run-with-graphics.sh rootproject/root "root --web=off"
podman run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm -it -v $HOME:/userhome --user $(id -u) $*
