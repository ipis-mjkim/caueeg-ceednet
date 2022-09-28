import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# from .utils import TimeElapsed

# __all__ = []


def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # pull the data
            x = sample_batched['signal']
            age = sample_batched['age']
            y = sample_batched['class_label']

            # mixed precision training if needed
            with autocast(enabled=config.get('mixed_precision', False)):
                # forward pass
                output = model(x, age)

                # loss function
                if config['criterion'] == 'cross-entropy':
                    s = F.log_softmax(output, dim=1)
                    loss = F.nll_loss(s, y)
                elif config['criterion'] == 'multi-bce':
                    y_oh = F.one_hot(y, num_classes=output.size(dim=1))
                    s = torch.sigmoid(output)
                    loss = F.binary_cross_entropy_with_logits(output, y_oh.float())
                elif config['criterion'] == 'svm':
                    s = output
                    loss = F.multi_margin_loss(output, y)
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

            # backward and update
            if config.get('mixed_precision', False):
                amp_scaler.scale(loss).backward()
                if 'clip_grad_norm' in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if 'clip_grad_norm' in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc


def train_mixup_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # load and mixup the mini-batched data
            x1 = sample_batched['signal']
            age1 = sample_batched['age']
            y1 = sample_batched['class_label']

            index = torch.randperm(x1.shape[0]).cuda()
            x2 = x1[index]
            age2 = age1[index]
            y2 = y1[index]

            mixup_alpha = config['mixup']
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            x = lam * x1 + (1.0 - lam) * x2
            age = lam * age1 + (1.0 - lam) * age2

            # mixed precision training if needed
            with autocast(enabled=config.get('mixed_precision', False)):
                # forward pass
                output = model(x, age)

                # loss function
                if config['criterion'] == 'cross-entropy':
                    s = F.log_softmax(output, dim=1)
                    loss1 = F.nll_loss(s, y1)
                    loss2 = F.nll_loss(s, y2)
                    loss = lam * loss1 + (1 - lam) * loss2
                elif config['criterion'] == 'multi-bce':
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                    y_oh = lam * y1_oh + (1.0 - lam) * y2_oh
                    s = torch.sigmoid(output)
                    loss = F.binary_cross_entropy_with_logits(output, y_oh)
                elif config['criterion'] == 'svm':
                    s = output
                    loss1 = F.multi_margin_loss(output, y1)
                    loss2 = F.multi_margin_loss(output, y2)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

            # backward and update
            if config.get('mixed_precision', False):
                amp_scaler.scale(loss).backward()
                if 'clip_grad_norm' in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if 'clip_grad_norm' in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc
