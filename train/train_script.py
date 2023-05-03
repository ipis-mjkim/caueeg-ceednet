import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import wandb
import pprint
from datetime import datetime

from .train_core import train_multistep, train_mixup_multistep
from optim import get_lr_scheduler
from .evaluate import check_accuracy
from .evaluate import check_accuracy_extended
from .evaluate import check_accuracy_multicrop
from .visualize import draw_lr_search_record
from .visualize import draw_roc_curve, draw_confusion

# __all__ = []


def learning_rate_search(
    config,
    model,
    train_loader,
    val_loader,
    preprocess_train,
    preprocess_test,
    trials,
    steps,
):
    learning_rate_record = []
    given_model_state = deepcopy(model.state_dict())

    # default learning rate range is set based on a minibatch size of 32
    min_log_lr = -3.2 + np.log10(config["minibatch"] * config.get("ddp_size", 1) / 32)
    max_log_lr = -6.0 + np.log10(config["minibatch"] * config.get("ddp_size", 1) / 32)

    for log_lr in np.linspace(min_log_lr, max_log_lr, num=trials):
        lr = 10**log_lr

        # recover the given  model state
        model.load_state_dict(deepcopy(given_model_state))
        # model.module.reset_weights() if config.get('ddp', False) else model.reset_weights()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type="constant_with_decay",  # constant for search
            iterations=config["total_samples"],
            warmup_steps=config["total_samples"],
        )
        amp_scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", False) else None

        tr_ms = train_multistep if config.get("mixup", 0) < 1e-12 else train_mixup_multistep
        loss, _ = tr_ms(
            model,
            train_loader,
            preprocess_train,
            optimizer,
            scheduler,
            amp_scaler,
            config,
            steps,
        )

        train_accuracy = check_accuracy(
            model,
            train_loader,
            preprocess_test,
            config,
            repeat=config.get("check_accuracy_repeat", 10),
        )
        val_accuracy = check_accuracy(
            model,
            val_loader,
            preprocess_test,
            config,
            repeat=config.get("check_accuracy_repeat", 10),
        )

        # Train accuracy for the final epoch is stored
        if np.isfinite(loss):
            learning_rate_record.append((log_lr, train_accuracy, val_accuracy))

        del optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # find the best starting point (if a tie occurs, average them)
    midpoints = np.array([(tr + vl) / 2 for _, tr, vl in learning_rate_record])
    induces = np.argwhere(midpoints == np.max(midpoints))
    best_log_lr = np.average(np.array([log_lr for log_lr, _, _ in learning_rate_record])[induces])

    # recover the given  model state
    model.load_state_dict(deepcopy(given_model_state))

    return 10**best_log_lr, learning_rate_record


def train_script(
    config,
    model,
    train_loader,
    val_loader,
    test_loader,
    multicrop_test_loader,
    preprocess_train,
    preprocess_test,
):
    # only the main process of DDP logs, evaluates, and saves
    main_process = config["ddp"] is False or config["device"].index == 0

    if main_process:
        print(f"\n{'*'*30} {'Configurations for Train':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    # load if using an existing model
    if config.get("init_from", None):
        init_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["init_from"]}/')
        checkpoint = torch.load(os.path.join(init_path, "checkpoint.pt"), map_location=config["device"])
        model.load_state_dict(checkpoint["model_state"])
        pprint.pprint(f'Load an existing model from {config["init_from"]}\n', width=120)

    # wandb init
    if main_process and config["use_wandb"]:
        if config.get("resume", None) is None:
            wandb.init(project=config.get("project", "noname"), reinit=True)
            wandb.run.name = wandb.run.id
        else:
            wandb.init(
                project=config.get("project", "noname"),
                id=config["resume"],
                resume="must",
            )

    # search an appropriate starting learning rate if needed
    if config.get("search_lr", False) and config.get("resume", None) is None:
        config["base_lr"], lr_search = learning_rate_search(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            preprocess_train=preprocess_train,
            preprocess_test=preprocess_test,
            trials=20,
            steps=500,
        )
        if main_process:
            draw_lr_search_record(lr_search, use_wandb=config["use_wandb"])

    # training iteration and other conditions
    config["base_lr"] = config["base_lr"] * config.get("search_multiplier", 1.0)
    config["iterations"] = round(config["total_samples"] / config["minibatch"] / config.get("ddp_size", 1))
    config["warmup_steps"] = max(round(config["iterations"] * config["warmup_ratio"]), config["warmup_min"])
    history_interval = max(config["iterations"] // config["num_history"], 1)

    # generate the trainers
    optimizer = optim.AdamW(model.parameters(), lr=config["base_lr"], weight_decay=config["weight_decay"])
    scheduler = get_lr_scheduler(
        optimizer,
        config["lr_scheduler_type"],
        iterations=config["iterations"],
        warmup_steps=config["warmup_steps"],
    )
    amp_scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", False) else None

    # local variable for training loop
    i_step = 0

    # load if resuming
    if config.get("resume", None):
        resume = config["resume"]
        save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["resume"]}/')
        checkpoint = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        config = checkpoint["config"]
        if main_process and config["use_wandb"]:
            wandb.config.update(config, allow_val_change=True)
        i_step = checkpoint["optimizer_state"]["state"][0]["step"]
        print(f"\n{'*'*30} {f'Training resumes from {resume}':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    if main_process:
        # update configurations
        if config["use_wandb"]:
            wandb.config.update(config)

            # track gradients and weights statistics if needed
            if config.get("watch_model", False):
                wandb.watch(model, log="all", log_freq=history_interval, log_graph=True)

        # directory to save
        run_name = wandb.run.name if config["use_wandb"] else datetime.now().strftime("%Y-%m%d-%H%M")
        if config["save_model"]:
            save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{run_name}/")
            os.makedirs(save_path, exist_ok=True)

    # train and validation routine
    while i_step < config["iterations"]:
        i_step += history_interval * config.get("ddp_size", 1)

        # train during 'history_interval' steps
        tr_ms = train_multistep if config.get("mixup", 0) < 1e-12 else train_mixup_multistep
        loss, train_acc = tr_ms(
            model=model,
            loader=train_loader,
            preprocess=preprocess_train,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            config=config,
            steps=history_interval,
        )
        # validation accuracy
        val_acc = check_accuracy(
            model,
            val_loader,
            preprocess_test,
            config,
            repeat=config.get("check_accuracy_repeat", 10) * 3,
        )

        # log
        if main_process:
            if config["use_wandb"]:
                wandb.log(
                    {
                        "Loss": loss,
                        "Train Accuracy": train_acc,
                        "Validation Accuracy": val_acc,
                        "Learning Rate": optimizer.state_dict()["param_groups"][0]["lr"],
                    },
                    step=i_step * config["minibatch"],
                )
            else:
                print(
                    f"{i_step:7>} / {config['iterations']:>7} iter - "
                    f"Loss: {loss:.4}, Train Acc.: {train_acc:.4}, Val. Acc.: {val_acc:.4}"
                )

            # save the model
            if config["save_model"]:
                checkpoint = {
                    "model_state": model.state_dict(),
                    "config": config,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
                os.replace(
                    os.path.join(save_path, "checkpoint.pt"),
                    os.path.join(save_path, "checkpoint.pt"),
                )

    # calculate the test accuracy
    if main_process:
        test_result = check_accuracy_extended(
            model=model,
            loader=test_loader,
            preprocess=preprocess_test,
            config=config,
            repeat=config.get("check_accuracy_repeat", 10) * 3,
        )
        test_acc, score, target, test_confusion, _ = test_result

        # calculate the test accuracy of the final model using multiple crop averaging
        multicrop_test_acc = check_accuracy_multicrop(
            model=model,
            loader=multicrop_test_loader,
            preprocess=preprocess_test,
            config=config,
            repeat=config.get("check_accuracy_repeat", 10) * 3,
        )

        # save the model
        if config["save_model"]:
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            os.replace(
                os.path.join(save_path, "checkpoint.pt"),
                os.path.join(save_path, "checkpoint.pt"),
            )

        # leave the message
        if config["use_wandb"]:
            wandb.log(
                {
                    "Test Accuracy": test_acc,
                    "Confusion Matrix (Array)": test_confusion,
                    "Multi-Crop Test Accuracy": multicrop_test_acc,
                }
            )
        else:
            print(f"\n{'*'*30} {run_name:^30} {'*'*30}\n")
            pprint.pprint(
                {
                    "Test Accuracy": test_acc,
                    "Confusion Matrix (Array)": test_confusion,
                    "Multi-Crop Test Accuracy": multicrop_test_acc,
                }
            )
            print(f"\n{'*'*92}\n")

        if config["draw_result"]:
            draw_roc_curve(
                score,
                target,
                config["class_label_to_name"],
                use_wandb=config["use_wandb"],
            )
            draw_confusion(
                test_confusion,
                config["class_label_to_name"],
                use_wandb=config["use_wandb"],
            )

        if config["use_wandb"]:
            wandb.run.finish()

    # release memory
    del optimizer, scheduler
    return
