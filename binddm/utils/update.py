from typing import Callable
from functools import partial

import torch

def original(pseudo_epoch, mu, lb) -> torch.Tensor:
    epoch = torch.tensor(pseudo_epoch)
    # \frac{\mu}{\mu+exp({\rm epoch}/\mu)}
    expect = mu/(mu+torch.exp(epoch)/mu)
    result = max(expect, lb)
    return result if isinstance(result, torch.Tensor) else torch.tensor(result)

def linear(pseudo_epoch, slope, lb) -> torch.Tensor:
    # max_pseudo_epoch = 200
    # y = 1 - slope * x \in [0,1]
    epoch = torch.tensor(pseudo_epoch)
    expect = 1 + slope * epoch
    result = max(expect, lb)
    return result if isinstance(result, torch.Tensor) else torch.tensor(result)

def circle(pseudo_epoch, r, lb):
    epoch = torch.tensor(pseudo_epoch)
    # t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
    arc_square = (r**2 - epoch**2).clamp(0)
    expect = torch.sqrt(arc_square) / r
    result = max(expect, lb)
    return result if isinstance(result, torch.Tensor) else torch.tensor(result)

def get_update_p_func(bias:dict) -> Callable[[int], torch.Tensor]:
    update_method = bias.get("update_method", "original")
    lower = bias.get("min_p", 0)
    if update_method == "original":
        mu = bias.get("mu", 12)
        fn = partial(original, mu=mu, lb=lower)
    elif update_method == "linear":
        slope = bias.get("slope", -0.005)
        fn = partial(linear, slope=slope, lb=lower)
    elif update_method == "circle":
        r = bias.get("r", 200)
        fn = partial(circle, r=r, lb=lower)
    else:
        ValueError(f"unknown update_method: {update_method} is specified")
    return fn