#!/bin/bash

CONFIG_FILE=""
GEOMETRY=""
TIMESTEP=1
OUTPUT_DIR="."

set -Eeuo pipefail
set -x

if [ -x "$(command -v docker)" ]; then
    echo "Docker installed"
    # command
else
    echo "Docker is not installed"
    echo "Install docker at https://docs.docker.com/get-docker/"
    exit 1
    # command
fi

while test $# -gt 0; do
	case "$1" in
		-h|--help)
            echo "help"
            exit 0
            ;;
        --config)
            shift
            if test $# -gt 0; then
                CONFIG_FILE=$1
            else
                echo "no config file specified"
                exit 1
            fi
            shift
            ;;
        -t|--timestep)
            shift
            if test $# -gt 0; then
                TIMESTEP=$1
            else
                echo "no timestep specified"
                exit 1
            fi
            shift
            ;;
        -g|--geometry)
            shift
            if test $# -gt 0; then
                GEOMETRY=$1
            else
                echo "no geometry specified"
                exit 1
            fi
            shift
            ;;
        -o|--output-dir)
            shift
            if test $# -gt 0; then
                OUTPUT_DIR=$1
            else
                echo "no output directory specified"
                exit 1
            fi
            shift
            ;;
    *)
      break
      ;;
  esac
done

# remove nlisim container if exist
docker stop nlisim_container || true
docker rm nlisim_container || true

# create container
docker create -a STDOUT --name nlisim_container nlisim

# copy config
if [ -z "$CONFIG_FILE" ]; then
    echo "No config file specified"
else
    docker cp "./$CONFIG_FILE" nlisim_container:./nlisim/config.ini
fi

if [ -z "$GEOMETRY" ]; then
    echo "No geometry file specified"
else
    docker cp "./$GEOMETRY" nlisim_container:./nlisim/"$GEOMETRY"
fi

# start and run container
docker start nlisim_container
docker exec nlisim_container nlisim run "$TIMESTEP"

# copy output
docker cp nlisim_container:./nlisim/output "$OUTPUT_DIR"

docker stop nlisim_container
docker rm nlisim_container
