#!/bin/bash

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to determine the package manager
get_package_manager() {
  if command_exists apt-get; then
    echo "apt-get"
  elif command_exists yum; then
    echo "yum"
  elif command_exists pacman; then
    echo "pacman"
  else
    echo "Unsupported package manager"
    exit 1
  fi
}

# Function to install a package
install_package() {
  local package_manager=$1
  local package=$2
  case $package_manager in
    apt-get)
      sudo apt-get update
      sudo apt-get install -y "$package"
      ;;
    yum)
      sudo yum install -y "$package"
      ;;
    pacman)
      sudo pacman -Syu --noconfirm "$package"
      ;;
  esac
}

# Function to install Conda
install_conda() {
  # Download the Conda installer
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    # Install Conda
    bash miniconda.sh -b -p $HOME/miniconda

    # Add Conda to the PATH
    export PATH="$HOME/miniconda/bin:$PATH"

    # Initialize Conda
    conda init

    # Cleanup
    rm miniconda.sh

    echo "Conda installation completed."
  }

# Function to download pretrained models
download_pretrained_models() {
  local repo_path=$1

    # Create the pretrained_models directory if it doesn't exist
    mkdir -p "$repo_path/pretrained_models"

    # Model specific settings
    model_name="VoiceCraft_gigaHalfLibri330M_TTSEnhanced_max16s"
    model_dir="$repo_path/pretrained_models/$model_name"
    mkdir -p "$model_dir"

    base_url="https://huggingface.co/pyp1/$model_name/resolve/main/"
    config_url="${base_url}config.json"
    model_safetensors_url="${base_url}model.safetensors"
    encodec_url="https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"

    config_path="$model_dir/config.json"
    model_safetensors_path="$model_dir/model.safetensors"
    encodec_path="$repo_path/pretrained_models/encodec_4cb2048_giga.th"

    # Download config.json
    if [ ! -f "$config_path" ]; then
      echo "Downloading config.json for $model_name..."
      wget -O "$config_path" "$config_url"
    fi

    # Download model.safetensors
    if [ ! -f "$model_safetensors_path" ]; then
      echo "Downloading model.safetensors for $model_name..."
      wget -O "$model_safetensors_path" "$model_safetensors_url"
    fi

    # Download encodec model
    if [ ! -f "$encodec_path" ]; then
      echo "Downloading encodec_4cb2048_giga.th..."
      wget -O "$encodec_path" "$encodec_url"
    fi
  }

# Function to install the VoiceCraft API
install_voicecraft_api() {
  local repo_path=$1

    # Create a conda environment named voicecraft_api
    conda create -n voicecraft_api python=3.9.16 -y

    # Activate the environment
    conda activate voicecraft_api

    # Install audiocraft
    pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft

    # Install PyTorch, etc.
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

    # Install the API requirements
    pip install -r "$repo_path/requirements.txt"

    # Install Montreal Forced Aligner
    conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 -y

    # Install Montreal Forced Aligner models
    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa

    # Download pretrained models
    download_pretrained_models "$repo_path"

    echo "VoiceCraft API installation completed."
  }

# Determine the package manager
package_manager=$(get_package_manager)

# Check if Conda is installed
if ! command_exists conda; then
  echo "Conda not found. Installing Conda..."
  install_conda
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi

# Check if Git is installed
if ! command_exists git; then
  echo "Git not found. Installing Git..."
  install_package $package_manager git
fi

# Check if Espeak is installed
if ! command_exists espeak-ng; then
  echo "Espeak not found. Installing Espeak..."
  install_package $package_manager espeak-ng
fi

# Check if FFmpeg is installed
if ! command_exists ffmpeg; then
  echo "FFmpeg not found. Installing FFmpeg..."
  install_package $package_manager ffmpeg
fi

# Check if the current directory is the VoiceCraft_API folder
if [ "${PWD##*/}" != "VoiceCraft_API" ]; then
  # If not in the VoiceCraft_API folder, clone the repository
  git clone https://github.com/lukaszliniewicz/VoiceCraft_API.git
  cd VoiceCraft_API
fi

# Install the VoiceCraft API
install_voicecraft_api "$PWD"

# Activate the conda environment and run the API
source activate voicecraft_api
python3 api.py
