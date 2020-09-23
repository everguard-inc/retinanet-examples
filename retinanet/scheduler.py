import math
from typing import NoReturn, Optional, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpWarmRestartsCooldown(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: float = 1,
        T_warmup: int = 0,
        eta_min: float = 1e-6,
        gamma: float = 1,
        eta_max: float = 0.1,
        last_epoch: int = -1,
        main_part: Optional[int] = None,
        eta_cooldown: Optional[float] = None,
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_warmup = T_warmup
        self.T_mult = T_mult

        # for decreasing eta_max
        self.cycle: int = 0
        self.gamma = gamma

        self.eta_max = eta_max
        self.base_eta_max = eta_max
        self.eta_min = eta_min
        self.eta_cooldown = eta_cooldown if eta_cooldown else eta_min

        # counters
        self.main_part = main_part if main_part else float("inf")
        self.total_iterations = -1
        self.T_cur = last_epoch

        super(CosineAnnealingWarmUpWarmRestartsCooldown, self).__init__(optimizer, last_epoch)
        self.eta_warmup: Sequence[float] = self.base_lrs

    def get_lr(self) -> Sequence[float]:
        # print(self.total_iterations, self.main_part, self.T_cur, self.base_lrs[0])
        # main part of learning: warmup + cosine annealing cycles
        if self.total_iterations < self.main_part:
            if self.T_cur <= 1:
                # if there is warmup, we start from very low learning rate else we take base learning rate
                if self.T_warmup > 0:
                    lrs = [self.eta_min for _ in self.base_lrs]
                    self.eta_warmup = lrs
                else:
                    lrs = self.base_lrs

            # warmup period, we linearly reach lr to eta_max
            elif self.T_cur <= self.T_warmup:
                lrs = [
                    (self.eta_max - base_warmup_lr) * self.T_cur / self.T_warmup + base_warmup_lr
                    for base_warmup_lr in self.eta_warmup
                ]

            # main cosine annealing cycle
            else:
                lrs = [
                    self.eta_min
                    + (self.eta_max - self.eta_min)
                    * (1 + math.cos(math.pi * (self.T_cur - self.T_warmup) / (self.T_i - self.T_warmup)))
                    / 2
                    for _ in self.base_lrs
                ]
        # cooldown
        else:
            lrs = [self.eta_cooldown for _ in self.base_lrs]
        return lrs

    def step(self, epoch: Optional[int] = None) -> NoReturn:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        self.total_iterations = self.total_iterations + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
