# Quick Setup Guide for Linux

## Install Conda
- Run
    ```
    wget -c https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
    bash Anaconda3-2023.07-2-Linux-x86_64.sh
    ```
- Then restart terminal to activate conda
- Create environment
    ```
    conda create -n odl_torch
    conda activate odl_torch
    ```
## Install dependencies
- Pytorch:
    `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- astra toolbox: `conda install -c astra-toolbox/label/dev astra-toolbox --yes`
- odl: `pip install git+https://github.com/odlgroup/odl.git`
- git: `git clone https://github.com/EricOldgren/KEX---CT-reconstruction.git`
- DONE