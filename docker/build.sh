#!/bin/bash

OFF='\033[0m'
RED='\033[0;31m'
GRN='\033[0;32m'
BLU='\033[0;34m'

BOLD=$(tput bold)
NORM=$(tput sgr0)

ERR="${RED}${BOLD}"
RRE="${NORM}${OFF}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

IMAGE_NAME="bz/mario-kart-a3c"
IMAGE_TAG="0.0.5"

if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "$IMAGE_NAME:$IMAGE_TAG"; then
    echo -e "${BOLD}Image ${BLU}${BOLD}$IMAGE_NAME${OFF} ${BOLD}with tag ${BLU}${BOLD}$IMAGE_TAG${OFF} ${BOLD}already exists.${RRE}"
    read -p "Enter a new tag for the image:" NEW_TAG
    IMAGE_TAG="$NEW_TAG"
    echo -e "Building new image ${GRN}${BOLD}$IMAGE_NAME:$IMAGE_TAG.${RRE}"
else
    echo -e "${GRN}${BOLD}Building default image $IMAGE_NAME and tag $IMAGE_TAG.${RRE}"
fi

sleep 2

docker build -t $IMAGE_NAME:$IMAGE_TAG -f $SCRIPT_DIR/Dockerfile $SCRIPT_DIR/..