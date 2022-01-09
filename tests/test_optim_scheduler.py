import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from simple_transformer.optim import Scheduler


def test_scheduler() -> None:
    model = nn.Sequential(nn.Linear(1, 1))
    optimizer = Adam(model.parameters(), lr=100) # <= This will be overridden by the scheduler

    dim_embed = 512
    warmup_steps = 4000
    scheduler = Scheduler(optimizer, dim_embed, warmup_steps)

    for step in range(100000):
        lr = dim_embed**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))

        assert np.allclose(scheduler.get_last_lr()[0], lr)
        
        optimizer.step()
        scheduler.step()

