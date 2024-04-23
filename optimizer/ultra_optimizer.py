from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_scheduler(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # linear warm-up
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        else:
            # cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
