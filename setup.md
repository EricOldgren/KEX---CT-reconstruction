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
    `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia --yes`
- astra toolbox: `conda install -c astra-toolbox/label/dev astra-toolbox --yes`
- odl: `pip install git+https://github.com/odlgroup/odl.git`
- matplotlib: `pip install matplotlib`
- sklearn: `pip install scikit-learn`
- git: `git clone https://github.com/EricOldgren/KEX---CT-reconstruction.git`
- DONE