import time
import wandb


def load_sweep_config(config):
    # load default configurations not selected by wandb.sweep
    cfg_sweep = dict()
    for k, v in config.items():
        if k not in [wandb_key.split(".")[-1] for wandb_key in wandb.config.keys()]:
            cfg_sweep[k] = v

    # load the selected configurations from wandb sweep with preventing callables from type-conversion to str
    for k, v in wandb.config.items():
        k = k.split(".")[-1]
        if k not in cfg_sweep:
            cfg_sweep[k] = v

    return cfg_sweep


class TimeElapsed(object):
    def __init__(self, header=""):
        self.header = header
        self.counter = 1
        self.start = time.time()

    def restart(self):
        self.start = time.time()
        self.counter = 1

    def elapsed_str(self):
        end = time.time()
        time_str = f"{self.counter:3d}> {end - self.start :.5f}"
        self.start = end
        self.counter += 1
        return time_str
