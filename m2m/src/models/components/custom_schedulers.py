import torch
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_epoch:
            return [
                base_lr * ((self.multiplier - 1) * self.last_epoch / self.total_epoch + 1)
                for base_lr in self.base_lrs
            ]
        if self.after_scheduler:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return self.base_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.after_scheduler and self.last_epoch >= self.total_epoch:
            self.after_scheduler.step(epoch - self.total_epoch)
        else:
            super().step(epoch)


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, after_scheduler=None):
        self.warmup = warmup  # warm up epochs
        self.max_num_iters = max_iters
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.max_num_iters:
            return [
                base_lr * self.get_lr_factor(epoch=self.last_epoch) for base_lr in self.base_lrs
            ]
        if self.after_scheduler:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return self.base_lrs

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = epoch / self.warmup  # Linear warmup
        else:
            # Using torch for cosine calculation, including torch.pi
            lr_factor = 0.5 * (
                1
                + torch.cos(torch.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup))
            )
        return lr_factor.item()  # Converting tensor to Python float

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.after_scheduler and self.last_epoch >= self.max_num_iters:
            self.after_scheduler.step(epoch - self.max_num_iters)
        else:
            super().step(epoch)
