#!/bin/bash

# Ensure that the script is called with the correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_image>"
    exit 1
fi

# Assign the first argument to PARAM1
PARAM1=$1

# Extract the image name without the extension
FOLDER_NAME=$(basename "$PARAM1" .png)

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stylegan

# Navigate to the Documents directory
cd ~/Documents

# Check if the folder exists, if not, create it
OUTDIR="./tesis/resultados/${FOLDER_NAME}"
if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
fi

# Run the docker command
sudo ./stylegan2-ada-pytorch/docker_run.sh python3 ./stylegan2-ada-pytorch/projector.py --outdir="$OUTDIR" --target="./tesis/fotos/${PARAM1}" --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

#python3 align_images.py raw_images/ aligned_images/ tengo que hacer esto antes