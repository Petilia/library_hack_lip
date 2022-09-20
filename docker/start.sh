#!/bin/bash

cd "$(dirname "$0")"
# cd ..

workspace_dir=$PWD

# -v $workspace_dir/../data:/home/docker_current/data:rw \
# -v $workspace_dir/../py_files:/home/docker_current/py_files:rw \

desktop_start() {
    xhost +local:
    docker run -it -d --rm \
        --gpus all \
        --ipc host \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name library_hack_lip \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $workspace_dir/../py_files:/home/docker_current/py_files:rw \
        -v /mnt/hdd8/petryashin_ie/library_challenge:/home/docker_current/datasets:rw \
        -v /mnt/hdd8/petryashin_ie/cache_lib_chal:/home/docker_current/.cache:rw \
        ${ARCH}/library_hack_lip:latest
    xhost -
}


main () {
    ARCH="$(uname -m)"

    if [ "$ARCH" = "x86_64" ]; then
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        arm_start;
    fi

}

main;
