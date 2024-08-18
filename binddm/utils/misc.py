import logging
import os
import random
import time
import sys
import gc
from typing import Union
from functools import reduce

import numpy as np
import torch
import yaml
from easydict import EasyDict


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def load_config(path, wrap=True):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return EasyDict(data) if wrap else data

def dump_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

PythonMinor = sys.version_info.minor
if PythonMinor < 10:
    import pdb
    bp = pdb.set_trace
else:
    bp = breakpoint


METHOD_SP = 0b1
METHOD_PSP = 0b10
METHOD_DICT = {
    "SP": METHOD_SP, # 0b0001
    "PSP": METHOD_PSP, # 0b0010
}

def method_resolve(method: Union[None, str, list]) -> int:
    if method == None:
        return 0
    elif isinstance(method, str):
        if method not in METHOD_DICT:
            raise ValueError(f"Invalid method: {method}")
        return METHOD_DICT[method]
    elif isinstance(method, list):
        return reduce(lambda x, y: x | y, (method_resolve(m) for m in method))
    else:
        raise TypeError(f"Invalid method: {method} with type: {type(method)}")


def __show_element_unit(el, unit:str) -> str:
    int_el = int(el)
    str_el = int_el if int_el == el else f"{el:.2f}"
    return f"{str_el}{unit}" if el else ""

def convert_seconds(duration: float) -> str:
    minute, second = divmod(duration, 60)
    hour, minute = divmod(minute, 60)
    day, hour = divmod(hour, 24)
    return ' '.join([
        __show_element_unit(v, u)
        for v,u in zip(
            [day, hour, minute, second],
            ["day", "hour", "min", "s"]
        )
    ])

def clean_grad(model:torch.nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            del p.grad
    gc_all()

def gc_all():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
