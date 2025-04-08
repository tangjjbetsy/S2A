import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from data.components.expression_tokenizer import ExpressionTok

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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


def visulization(preds, gts, style, save_dir, perf_name):
    preds = preds.T.tolist()
    gts = gts.T.tolist()
    style = PERFROMER[style]
    org_style = PERFROMER[int(perf_name.split("_")[2])]

    out_feats = ["Velocity", "IOI", "Duration"]
    sns.set_theme(style="darkgrid")

    # fig.set_size_inches(13,18)

    num_notes = len(preds[0])
    x = np.linspace(0, num_notes - 1, num=num_notes)

    for i in range(3):
        fig, axes = plt.subplots(2, sharex=True)
        fig.set_size_inches(15, 10)
        if org_style == style:
            pred_name = "perf_p"
        else:
            pred_name = f"perf_p_{style}"

        gt_name = f"perf_{org_style}"

        df = pd.DataFrame(columns=["time", gt_name, pred_name])
        df["time"] = x
        df[gt_name] = None
        df[pred_name] = None
        df.loc[0 : len(gts[i]) - 1, gt_name] = gts[i]
        df.loc[0 : len(preds[i]) - 1, pred_name] = preds[i]
        df["residual"] = df[gt_name] - df[pred_name]
        df_melt = df.melt(
            id_vars="time",
            value_vars=[gt_name, pred_name],
            var_name="performance",
            value_name=out_feats[i],
        )

        ax1 = sns.lineplot(
            x="time", y=out_feats[i], data=df_melt, hue="performance", alpha=0.5, ax=axes[0]
        )
        ax1.set_title(f"Comparison: {out_feats[i]}")

        ax2 = sns.lineplot(x="time", y="residual", data=df, alpha=0.5, ax=axes[1])
        ax2.set_xticks(
            ticks=df_melt["time"].tolist()[::64],
            labels=[int(i) for i in df_melt["time"].tolist()[::64]],
        )
        ax2.set_xlabel("notes")
        if out_feats[i] == "Velocity":
            ax2.set_ylim(-25, 25)
        elif out_feats[i] == "IOI":
            ax2.set_ylim(-100, 100)
        else:
            ax2.set_ylim(-200, 200)
        if (
            os.path.isdir(
                f"{save_dir}/predictions/{perf_name.split('_')[0]}_{org_style}/{out_feats[i]}"
            )
            is False
        ):
            os.makedirs(
                f"{save_dir}/predictions/{perf_name.split('_')[0]}_{org_style}/{out_feats[i]}"
            )
        plt.savefig(
            f"{save_dir}/predictions/{perf_name.split('_')[0]}_{org_style}/{out_feats[i]}/{perf_name.split('_')[1]}_{style}_compare.png"
        )
        plt.clf()


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


def reconstruct_midi(predictions, cfg, tokenizer):
    indexes = get_indexes_for_same_perf(predictions)
    for key, value in indexes.items():
        value_keys = list(sorted(value.keys(), key=lambda x: int(x.split("_")[1])))
        tokens = []
        for k in range(6):
            for i in range(len(value_keys)):
                perf = predictions[value[value_keys[i]][0]][0][k]
                perf = torch.stack(perf, dim=2).squeeze()[value[value_keys[i]][1]]
                score = predictions[value[value_keys[i]][0]][1][value[value_keys[i]][1]]
                mask = predictions[value[value_keys[i]][0]][2][value[value_keys[i]][1]]
                gt = predictions[value[value_keys[i]][0]][3][value[value_keys[i]][1]]
                org_style = predictions[value[value_keys[i]][0]][4][value[value_keys[i]][1]]
                style = k
                perf = mask.unsqueeze(-1).repeat(1, 3) * perf

                for j in range(len(cfg.data.output_features)):
                    bounds = cfg.model.net.bert.feature_boundaries[cfg.data.output_features[j]]
                    perf[:, [j]] = clip_feature(perf[:, [j]], bounds)

                alignment_tokens = (
                    torch.cat(
                        [
                            score[:, [0]],  # Pitch
                            perf[:, [0]],  # PVel
                            perf[:, [2]],  # PDur
                            perf[:, [1]],  # PIOI
                            score[:, [4, 5]],  # Fake-PPos, Bar
                            score[:, 1:],
                        ],  # Score
                        dim=-1,
                    )
                    .int()
                    .tolist()
                )

                # visulization(perf, gt, style, cfg.paths.output_dir, f"{value_keys[i]}_{org_style}")

                if i == 0:
                    tokens = alignment_tokens
                else:
                    tokens += alignment_tokens

            if os.path.isdir(cfg.paths.output_dir + "/predictions/") is False:
                os.makedirs(cfg.paths.output_dir + "/predictions/")
            if os.path.isdir(cfg.paths.output_dir + "/scores/") is False:
                os.makedirs(cfg.paths.output_dir + "/scores/")

            tokenizer.align_tokens_to_midi(
                [tokens],
                cfg.paths.output_dir + "/predictions/" + key + f"_{org_style}_{style}.mid",
                cfg.paths.output_dir + "/scores/" + key + ".mid",
                from_predictions=True,
            )


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

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

    log.info("Starting evaluating!")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    torch.save(predictions, cfg.paths.output_dir + "/predictions.pt")
    reconstruct_midi(predictions, cfg, tokenizer)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
