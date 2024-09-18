#!/bin/bash

########################################## clone_repos.sh ##########################################
####################################################################################################

if [[ -z "$1" ]]; then
    echo "Error: Please provide a download directory!"
    echo "Usage: $0 <download_directory> [--model_list model_1,model_2,...]"
    exit 1
fi

download_directory="$1"

declare -A repos=(
    ["Open-Sora"]="git@github.com:hpcaitech/Open-Sora.git"
    ["VADER"]="https://github.com/mihirp1998/VADER.git"
    ["VideoCrafter"]="git@github.com:AILab-CVC/VideoCrafter.git"
    ["StableVideoDiffusion"]="git@github.com:Stability-AI/generative-models.git"
    ["ModelScope"]="https://modelscope.cn/models/iic/text-to-video-synthesis"
    ["InstructVideo"]="https://github.com/ali-vilab/VGen.git"
)

# process --model_list
model_list=()
if [[ "$2" == "--model_list" ]]; then
    IFS=',' read -r -a model_list <<< "$3"  
fi

if [[ ${#model_list[@]} -eq 0 ]]; then
    model_list=("${!repos[@]}")  # default to download all
fi

for model in "${model_list[@]}"; do
    if [[ -z "${repos[$model]}" ]]; then
        echo "Error: Model '$model' is not in the list of available models."
        echo "Available models: ${!repos[*]}"
        exit 1
    fi
done

## process download dir
mkdir -p "$download_directory"
cd "$download_directory" || exit 1

for model in "${model_list[@]}"; do
    repo_url="${repos[$model]}"
    
    if [[ -d "$model" ]]; then
        echo "Repository $model already exists, skipping clone."
    else
        echo "Cloning $model ..."
        git clone "$repo_url"
        if [[ $? -ne 0 ]]; then
            echo "Failed to clone $model, please check the repository URL or network connection."
            exit 1
        fi
    fi
done

echo "All specified model repositories have been cloned."