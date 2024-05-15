#!/bin/sh -e
#give permission
xhost +local:root

$*

#reset permission
xhost -local:root
