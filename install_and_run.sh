#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
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

# Function to install Git
install_git() {
    sudo apt-get update
    sudo apt-get install git -y
    echo "Git installation completed."
}

# Function to install Espeak
install_espeak() {
    sudo apt-get update
    sudo apt-get install espeak-ng -y
    echo "Espeak installation completed."
}

# Function to install FFmpeg
install_ffmpeg() {
    sudo apt-get update
    sudo apt-get install ffmpeg -y
    echo "FFmpeg installation completed."
}

# Function to install the VoiceCraft API
install_voicecraft_api() {
    # Clone the VoiceCraft API repository
    git clone https://github.com/lukaszliniewicz/VoiceCraft_API.git

    # Change into the repository directory
    cd VoiceCraft_API

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
    wget -P pretrained_models https://huggingface.co/pyp1/VoiceCraft/resolve/main/giga830M.pth?download=true

    echo "VoiceCraft API installation completed."
}

# Check if Conda is installed
if ! command_exists conda; then
    echo "Conda not found. Installing Conda..."
    install_conda
    source "$HOME/miniconda/etc/profile.d/conda.sh"
fi

# Check if Git is installed
if ! command_exists git; then
    echo "Git not found. Installing Git..."
    install_git
fi

# Check if Espeak is installed
if ! command_exists espeak-ng; then
    echo "Espeak not found. Installing Espeak..."
    install_espeak
fi

# Check if FFmpeg is installed
if ! command_exists ffmpeg; then
    echo "FFmpeg not found. Installing FFmpeg..."
    install_ffmpeg
fi

# Check if the repo folder exists
if [ -d "VoiceCraft_API" ]; then
    # If the repo folder exists, activate the conda environment and run the API
    cd VoiceCraft_API
    source activate voicecraft_api
    python3 api.py
else
    # If the repo folder doesn't exist, install the VoiceCraft API
    install_voicecraft_api
fi
