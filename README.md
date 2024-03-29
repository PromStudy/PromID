# PromID: A deep learning-based tool to identify promoters

## Installation
PromID can be installed from the [github repository](https://github.com/PromStudy/PromID.git):
```sh
git clone https://github.com/PromStudy/PromID.git
cd PromID
pip install .
```
PromID requires ```tensorflow>=1.7.0```, the GPU version is highly recommended.
First, install Conda and create the environment:
```sh
conda create -n PromID python=3.6
conda activate PromID
```
Next, install tensorflow:
```sh
conda install -c conda-forge tensorflow==1.7.0
```
OR
```sh
conda install -c conda-forge tensorflow-gpu==1.7.0
```
for the GPU version. If that does not work, try removing "-c conda-forge".  
If you chose the GPU version, please also install CUDA9 and cuDNN7:
```sh
conda install cudatoolkit=9.0
conda install cudnn=7.1.2=cuda9.0_0
```
## Usage
PromID can be run from the command line:
```sh
promid -I hg19.fa -O hg19_promoters.bed -C chr20
```
Required parameters:
 - ```-I```: Input fasta file.
 - ```-O```: Output bed file.

Optional parameters:
 - ```-D```: Minimum soft distance between the predicted TSS, defaults to 1000.
 - ```-C```: Comma separated list of chromosomes to use for promoter prediction, defaults to all.
 - ```-T```: Decision threshold for the prediction model, defaults to 0.5.
