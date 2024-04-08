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

# Function to install the VoiceCraft API
install_voicecraft_api() {
    # Create a conda environment named voicecraft_api
    conda create -n voicecraft_api python=3.9.16 -y
    # Activate the environment
    source activate voicecraft_api
    # Install audiocraft
    pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
    # Install PyTorch, etc.
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    # Install the API requirements
    pip install -r requirements.txt
    # Install Montreal Forced Aligner
    conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 -y
    # Install Montreal Forced Aligner models
    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa
    # Download the model and encoder
    wget -P pretrained_models https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th
    wget -P pretrained_models https://huggingface.co/pyp1/VoiceCraft/resolve/main/gigaHalfLibri330M_TTSEnhanced_max16s.pth
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
install_voicecraft_api

# Activate the conda environment and run the API
source activate voicecraft_api
python3 api.py
