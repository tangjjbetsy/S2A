from typing import Any, Dict, Tuple

import torch
from gradnorm_pytorch import GradNormLossWeighter
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

from .components import ExpressionDWTLoss, ExpressionL1Loss


class S2PLitModule(LightningModule):
    """A `LightningModule` for score to performance task."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool,
        warm_up_step: int,
        normalize_loss: bool,
        penalty_outliers: bool,
        dynamic_weights: bool,
        loss_type: str,
    ) -> None:
        """Initialize a `S2PLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        if self.hparams.dynamic_weights:
            backbone_parameter = self.net.out_linear.weight
            self.loss_weighter = GradNormLossWeighter(
                num_losses=len(net.bert.output_features),
                learning_rate=1e-3,
                restoring_force_alpha=0.0,  # 0. is perfectly balanced losses, while anything greater than 1 would account for the relative training rates of each loss. in the paper, they go as high as 3.
                grad_norm_parameters=backbone_parameter,
            )

        # other features
        self.output_features = net.bert.output_features
        self.output_feature_boundaries = [
            net.bert.feature_boundaries[feat] for feat in self.output_features
        ]

        # loss function
        self.expression_criterion = ExpressionL1Loss(
            self.output_features,
            self.output_feature_boundaries,
            self.hparams.normalize_loss,
            self.hparams.penalty_outliers,
        )
        self.dwt_criterion = ExpressionDWTLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_dwt_loss = MeanMetric()
        self.valid_dwt_loss = MeanMetric()
        self.test_dwt_loss = MeanMetric()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        y_ = self.net(x, mask, style)
        return y_

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.val_loss.reset()
        self.train_dwt_loss.reset()
        self.valid_dwt_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, mask, y, style, idx = batch

        y_ = self.forward(x, mask, style)
        exp_loss = self.expression_criterion(y_, y, mask)
        dwt_loss = self.dwt_criterion(y_, y)

        return exp_loss, dwt_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt = self.optimizers()
        lrs = self.lr_schedulers()
        exp_loss, dwt_loss = self.model_step(batch)
        self.train_dwt_loss.update(dwt_loss)
        self.log(
            "train/dwt_loss", self.train_dwt_loss, on_step=True, on_epoch=True, prog_bar=False
        )

        l1_loss = [exp_loss[feat]["l1_loss"] for feat in exp_loss.keys()]
        for i in range(len(self.output_features)):
            self.log(
                f"train/l1_loss_{self.output_features[i]}",
                l1_loss[i],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        opt.zero_grad()
        if (self.hparams["loss_type"] == "l1") and (self.hparams.dynamic_weights):
            self.loss_weighter.backward(l1_loss, retain_graph=True)
            total_loss = torch.mul(torch.stack(l1_loss), self.loss_weighter.loss_weights).sum()

            self.train_loss.update(total_loss)
            self.log("train/total_loss", self.train_loss, prog_bar=True)
            for i in range(len(self.output_features)):
                self.log(
                    f"train/weight_{self.output_features[i]}",
                    self.loss_weighter.loss_weights[i],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )

        elif (self.hparams["loss_type"] == "l1") and (self.hparams.dynamic_weights is False):
            bounds = torch.FloatTensor([b[1] for b in self.output_feature_boundaries])
            loss_weights = bounds / torch.sum(bounds)
            total_loss = torch.mul(torch.stack(l1_loss), loss_weights.to(self.device)).sum()

            self.manual_backward(total_loss)

            self.train_loss.update(total_loss)
            self.log("train/total_loss", self.train_loss, prog_bar=True)

        else:
            self.manual_backward(dwt_loss.mean())

        opt.step()
        lrs.step()
        return

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log("train/total_loss", self.train_loss.compute(), prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        exp_loss, dwt_loss = self.model_step(batch)
        self.valid_dwt_loss.update(dwt_loss)
        self.log(
            "valid/dwt_loss", self.valid_dwt_loss, on_step=True, on_epoch=True, prog_bar=False
        )

        l1_loss = [exp_loss[feat]["l1_loss"] for feat in exp_loss.keys()]
        for i in range(len(self.output_features)):
            self.log(
                f"valid/l1_loss_{self.output_features[i]}",
                l1_loss[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if self.hparams.dynamic_weights:
            total_loss = torch.mul(torch.stack(l1_loss), self.loss_weighter.loss_weights).sum()
        else:
            if self.hparams["loss_type"] == "l1":
                bounds = torch.FloatTensor([b[1] for b in self.output_feature_boundaries])
                loss_weights = bounds / torch.sum(bounds)
                total_loss = torch.mul(torch.stack(l1_loss), loss_weights.to(self.device)).sum()
            else:
                total_loss = dwt_loss.mean()

        # update and log metrics
        self.val_loss.update(total_loss)
        self.log("valid/total_loss", self.val_loss, prog_bar=True)
        return

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.log("valid/total_loss", self.val_loss.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        exp_loss, dwt_loss = self.model_step(batch)
        self.test_dwt_loss.update(dwt_loss)
        self.log("test/dwt_loss", self.test_dwt_loss, on_step=True, on_epoch=True, prog_bar=False)

        l1_loss = [exp_loss[feat]["l1_loss"] for feat in exp_loss.keys()]

        if self.hparams.dynamic_weights:
            total_loss = torch.mul(torch.stack(l1_loss), self.loss_weighter.loss_weights).sum()
        else:
            if self.hparams["loss_type"] == "l1":
                bounds = torch.FloatTensor([b[1] for b in self.output_feature_boundaries])
                loss_weights = bounds / torch.sum(bounds)
                total_loss = torch.mul(torch.stack(l1_loss), loss_weights.to(self.device)).sum()
            else:
                total_loss = dwt_loss.mean()

        # update and log metrics
        self.test_loss.update(total_loss)
        for i in range(len(self.output_features)):
            self.log(
                f"test/l1_loss_{self.output_features[i]}",
                l1_loss[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        self.log("test/total_loss", self.test_loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/total_loss", self.test_loss.compute(), prog_bar=True)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, mask, y, style, idx = batch
        y_ = self.forward(x, mask, style)
        return y_, x, mask, y, style, idx

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.99999**epoch

            return lr_scale

        scheduler = self.hparams.scheduler(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    _ = S2PLitModule(None, None, None, None)
