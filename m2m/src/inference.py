import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from dtw import *
from tqdm import tqdm
from data.components.expression_tokenizer import ExpressionTok
from data.components.tools import pad_sequence_with_attention
import utils.sampling as sampling


# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

PERFROMER = [
    "Alfred Brendel",
    "Claudio Arrau",
    "Daniel Barenboim",
    "Friedrich Gulda",
    "Sviatoslav Richter",
    "Wilhelm Kempff",
]

log = RankedLogger(__name__, rank_zero_only=True)

def clip_feature(tokens, bounds):
    tokens[tokens > bounds[1]] = bounds[1]
    tokens[tokens < bounds[0]] = bounds[0]
    return tokens

def get_indexes_for_same_perf(predictions):
    # find segments for one piece
    indexes = {}
    idx = [batch[5] for batch in predictions]
    flat_idx = [item for sublist in idx for item in sublist]
    ids = list({i.split("_")[0] for i in flat_idx})
    indexes = dict()
    for i in ids:
        indexes[i] = dict()
        for k, batch in enumerate(predictions):
            for j, name in enumerate(batch[5]):
                if i in name:
                    indexes[i][name] = (k, j)

    return indexes

def reconstruct_midi(seqs, pred_seqs, cfg, tokenizer):
    for i, (seq, pred_seq) in enumerate(zip(seqs, pred_seqs)):
        perf = torch.stack(pred_seq, dim=2).squeeze()
     
        for j in range(len(cfg.data.output_features)):
            bounds = cfg.model.net.bert.feature_boundaries[cfg.data.output_features[j]]
            perf[:, [j]] = clip_feature(perf[:, [j]], bounds)

        alignment_tokens = (
            torch.cat(
                [
                    seq[:, [0]],  # Pitch
                    perf[:, [0]],  # PVel
                    perf[:, [2]],  # PDur
                    perf[:, [1]],  # PIOI
                    seq[:, [4, 5]],  # Fake-PPos, Bar
                    seq[:, 1:],
                ],  # Score
                dim=-1,
            )
            .int()
            .tolist()
        )
                        
        if i == 0:
            tokens = alignment_tokens
        else:
            tokens += alignment_tokens

    tokens = np.array(tokens)
    # write out the tokens to text file
    with open(cfg.paths.output_dir + "/output.txt", "w") as f:
        for token in tokens:
            f.write(" ".join([str(t) for t in token]) + "\n")
    tokenizer.align_tokens_to_midi(
        [tokens],
        cfg.paths.output_dir + "/output.mid",
        cfg.paths.output_dir + "/score.mid",
        from_predictions=True,
    )
            
def process_midi(midi_path, tokenizer):
    tokens = tokenizer(midi_path)[0].ids
    category_length = len(tokens[0])
    bos_row = [[1] * category_length] # BOS_ID = 1
    eos_row = [[2] * category_length] # EOS_ID = 2
    sequence = bos_row + tokens + eos_row
    sequences, masks = pad_sequence_with_attention(sequence, 256, True)
    return sequences, masks

@task_wrapper
def inference(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path
    assert cfg.midi_path
    assert cfg.style is not None


    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Loading tokenizer!")
    tokenizer = ExpressionTok(params=Path(cfg.tokenizer_path))
    seqs, masks = process_midi(cfg.midi_path, tokenizer)
    
    if torch.cuda.is_available():
        seqs = torch.LongTensor(np.stack(seqs, axis=0)).cuda()
        masks = torch.FloatTensor(np.stack(masks, axis=0)).cuda()
        styles = torch.LongTensor([cfg.style] * seqs.size(0)).cuda()
        model.cuda()
    else:
        seqs = torch.LongTensor(np.stack(seqs, axis=0))
        masks = torch.FloatTensor(np.stack(masks, axis=0))
        styles = torch.LongTensor([cfg.style] * seqs.size(0))

    log.info("Starting inferencing!")
    ckpt = torch.load(cfg.ckpt_path, map_location=model.device, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    pred_seqs = []
    for i, (seq, mask, style) in enumerate(zip(seqs, masks, styles)):
        y_ = model(seq.unsqueeze(0), mask.unsqueeze(0), style.unsqueeze(0))
        y_[0] = sampling.sampling(y_[0], t=1)
        y_[1] = sampling.sampling(y_[1], t=1)
        y_[2] = sampling.sampling(y_[2], t=1)
        
        pred_seqs.append(y_)
    
    reconstruct_midi(seqs, pred_seqs, cfg, tokenizer)

    return None, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    inference(cfg)


if __name__ == "__main__":
    main()
