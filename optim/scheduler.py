import math
from torch.optim import Optimizer
from torch.optim import lr_scheduler

lr_scheduler_list = [
    "constant_with_decay",
    "constant_with_twice_decay",
    "transformer_style",
    "cosine_decay_with_warmup_half",
    "cosine_decay_with_warmup_one_and_half",
    "cosine_decay_with_warmup_two_and_half",
    "linear_decay_with_warmup",
]


def get_constant_with_decay_scheduler(optimizer: Optimizer, iterations: int, last_epoch: int = -1):
    return lr_scheduler.StepLR(optimizer, step_size=round(iterations * 0.8), gamma=0.1, last_epoch=last_epoch)


def get_constant_with_twice_decay_scheduler(optimizer: Optimizer, iterations: int, last_epoch: int = -1):
    return lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(iterations * 0.7), round(iterations * 0.9)], gamma=0.1, last_epoch=last_epoch
    )


def get_transformer_style_scheduler(optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
    def transformer_style_lambda(step: int):
        return min(math.sqrt(warmup_steps) / max(1.0, math.sqrt(step)), step / max(1.0, float(warmup_steps)))

    return lr_scheduler.LambdaLR(optimizer, transformer_style_lambda, last_epoch=last_epoch)


def get_cosine_decay_with_warmup(
    optimizer: Optimizer, warmup_steps: int, iterations: int, cycles: float = 0.5, last_epoch: int = -1
):
    def cosine_decay_with_warmup_lambda(step):
        if step <= warmup_steps:
            return step / max(1.0, float(warmup_steps))
        period = (step - warmup_steps) / max(1.0, float(iterations - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * period)))

    return lr_scheduler.LambdaLR(optimizer, cosine_decay_with_warmup_lambda, last_epoch)


def get_linear_decay_with_warmup(optimizer: Optimizer, warmup_steps: int, iterations: int, last_epoch: int = -1):
    def linear_decay_with_warmup(step: int):
        if step <= warmup_steps:
            return step / max(1.0, float(warmup_steps))
        return max(0.0, (iterations - step) / max(1.0, float(iterations - warmup_steps)))

    return lr_scheduler.LambdaLR(optimizer, linear_decay_with_warmup, last_epoch)


def get_lr_scheduler(
    optimizer: Optimizer, scheduler_type: str, iterations: int, warmup_steps: int, last_epoch: int = -1
):
    if scheduler_type == "constant_with_decay":
        return get_constant_with_decay_scheduler(optimizer=optimizer, iterations=iterations, last_epoch=last_epoch)
    elif scheduler_type == "constant_with_twice_decay":
        return get_constant_with_twice_decay_scheduler(
            optimizer=optimizer, iterations=iterations, last_epoch=last_epoch
        )
    elif scheduler_type == "transformer_style":
        return get_transformer_style_scheduler(optimizer=optimizer, warmup_steps=warmup_steps, last_epoch=last_epoch)
    elif scheduler_type == "cosine_decay_with_warmup_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=0.5, last_epoch=last_epoch
        )
    elif scheduler_type == "cosine_decay_with_warmup_one_and_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=1.5, last_epoch=last_epoch
        )
    elif scheduler_type == "cosine_decay_with_warmup_two_and_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=2.5, last_epoch=last_epoch
        )
    elif scheduler_type == "linear_decay_with_warmup":
        return get_linear_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, last_epoch=last_epoch
        )
    else:
        raise ValueError(
            f"ERROR: get_lr_scheduler(scheduler_type) input is not understandable: {scheduler_type}. "
            f"Check the input value again: {lr_scheduler_list}"
        )
