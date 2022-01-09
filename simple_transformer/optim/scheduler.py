import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(_LRScheduler):
    """ This scheduler ignores the initial learning rate in the optimizer
    and overrides it with the calculated value.
    """
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.calc_lr = lambda step: calc_lr(step, dim_embed, warmup_steps)
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


if __name__=='__main__':
    # Test the learning rate calculation
    import numpy as np
    import matplotlib.pyplot as plt

    steps = np.arange(1, 10000)
    lrs = [calc_lr(step, 512, 4000) for step in steps]
    plt.plot(steps, lrs)
    plt.savefig('scheduler.png')
