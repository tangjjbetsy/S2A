<div align="center">

# RHEPP-S2P: Reconstruct Human Expressiveness in Piano Performances (with Music Scores)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/tangjjbetsy/RHEPP-Transformer-S2P"><img alt="Template" src="https://img.shields.io/badge/-RHEPP--Transformer--S2P-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This is the repository implementing and optimising the paper "RHEPP-Transformer: Reconstructing Human Expressiveness in Piano Performances with a Transformer Network" with the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). We retrained the model with real scores instead of the transcribed scores. Steps for inferencing with the current model will be released soon.

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/tangjjbetsy/RHEPP-Score
cd RHEPP-Score

# create conda environment and install dependencies
conda env create -f environment.yaml -n s2p

# activate conda environment
conda activate s2p
```

## How to run

Please refer to [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) for a better mannul of how to use this repo. This repo strictly follows the template design.

### Training

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# Using Probablistic Loss (Cross Entropy Loss)
python src/train.py experiment=s2p_bert_class
# Using L1 Loss, we remove the activation function used in the original paper
python src/train.py experiment=s2p_bert_no_activation
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

### Evaluation

Evaluate the model with the test set. Inferenced midis will be saved in the `logs/${EXP_DIR}/runs/${RUN_NAME}/predictions`. Note that the test set consists of processed alignment files not original scores

```bash
# Please configure the evaluation in configs/eval.yaml
python src/eval.py ckpt_path = PATH_TO_SAVED_MODEL
```

### Inference

TBC

### Scripts

The `data.sh` was created for preparing the dataset. `run.sh` was used to train the model on slurm.
