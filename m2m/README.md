# Midi-to-Midi Model

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/tangjjbetsy/RHEPP-Transformer-S2P"><img alt="Template" src="https://img.shields.io/badge/-RHEPP--Transformer--S2P-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div> -->

## Description
This is the M2M model implemented with the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). The model is trained with aligned scores and performances from ATEPP dataset.

## Installation

```bash
# create conda environment and install dependencies
conda env create -f environment.yaml

# activate conda environment
conda activate m2m
```

## Dataset & Checkpoints
Please download the [ATEPP-s2a](https://zenodo.org/records/15524693/files/ATEPP-s2a.zip) (midi files & aligned data) dataset and [checkpoints](https://zenodo.org/records/15524693/files/m2m.zip).

## Running
Please refer to [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) for more details on how to use this repo. This repo strictly follows the template design.

Working Directory: `m2m/`

### Data Preparation
```bash
./scripts/data.sh
```

### Training
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```bash
# Using Probablistic Loss (Cross Entropy Loss)
python src/train.py experiment=s2p_bert_class
```
You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

### Evaluation

Evaluate the model with the test set with the checkpoint `logs/s2p_bert_class/2024-05-22_02-26-21/checkpoints/epoch_1515.ckpt` or any other checkpoint you have trained.

```bash
# Please configure the evaluation in configs/eval.yaml
CHECKPOINT=logs/s2p_bert_class/2024-05-22_02-26-21/checkpoints/epoch_1515.ckpt
python src/eval.py ckpt_path=$CHECKPOINT
```

The script will provided the matrix results in the following format:
```bash
----------Performance-Wise Evaluation Results----------

---------vel----------
corr: mean → 0.8301, h → 0.0170
dtwd: mean → 4.2634, h → 0.1149
kld: mean → 0.0179, h → 0.0013

---------ioi----------
corr: mean → 0.9906, h → 0.0029
dtwd: mean → 7.5996, h → 0.6357
kld: mean → 0.0003, h → 0.0001

---------dur----------
corr: mean → 0.7549, h → 0.0171
dtwd: mean → 29.2187, h → 3.6145
kld: mean → 0.1895, h → 0.0099

----------Segment-Wise Evaluation Results----------

---------vel----------
corr: mean → 0.6654, h → 0.0106
dtwd: mean → 4.4432, h → 0.0363
kld: mean → 0.0170, h → 0.0004

---------ioi----------
corr: mean → 0.9346, h → 0.0062
dtwd: mean → 7.9537, h → 0.2493
kld: mean → 0.0003, h → 0.0000

---------dur----------
corr: mean → 0.6654, h → 0.0117
dtwd: mean → 30.5040, h → 1.2898
kld: mean → 0.1813, h → 0.0031
```

The corresponding predicted midis will be saved to the `logs/s2p_bert_class_evaluate/runs/${RUN_NAME}/predictions` folder, and the score midis will be saved in `logs/s2p_bert_class_evaluate/runs/${RUN_NAME}/scores`.