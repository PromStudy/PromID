# PromID: A deep learning-based tool to identify promoters

## Installation
PromID can be installed from the [github repository](https://github.com/PromStudy/PromID.git):
```sh
git clone https://github.com/PromStudy/PromID.git
cd PromID
pip install .
```
PromID requires ```tensorflow>=1.7.0```, the GPU version is highly recommended.

## Usage
PromID can be run from the command line:
```sh
promid -I hg19.fa -O hg19_promoters.bed
```
Required parameters:
 - ```-I```: Input fasta file.
 - ```-O```: Output bed file.

Optional parameters:
 - ```-D```: Minimum soft distance between the predicted TSS, defaults to 1000.
 - ```-C```: Comma separated list of chromosomes to use for promoter prediction, defaults to all.
 - ```-T1```: Decision threshold for the scan model, defaults to 0.2.
 - ```-T2```: Decision threshold for the prediction model, defaults to 0.5.