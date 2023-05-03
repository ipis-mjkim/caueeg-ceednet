import numpy as np
import torch
import torch.nn.functional as F

# __all__ = []


@torch.no_grad()
def compute_embedding(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]
    output = model.compute_feature_embedding(x, age, target_from_last=1)

    return output


@torch.no_grad()
def estimate_score(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]
    output = model(x, age)

    if config["criterion"] == "cross-entropy":
        score = F.softmax(output, dim=1)
    elif config["criterion"] == "multi-bce":
        score = torch.sigmoid(output)
    elif config["criterion"] == "svm":
        score = output
    else:
        raise ValueError(f"estimate_score(): cannot parse config['criterion']={config['criterion']}.")
    return score


def calculate_confusion_matrix(pred, target, num_classes):
    N = target.shape[0]
    C = num_classes
    confusion = np.zeros((C, C), dtype=np.int32)

    for i in range(N):
        r = target[i]
        c = pred[i]
        confusion[r, c] += 1
    return confusion


def calculate_class_wise_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    accuracy = np.zeros((n_classes,))
    sensitivity = np.zeros((n_classes,))
    specificity = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp

        accuracy[c] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[c] = tp / (tp + fn)
        specificity[c] = tn / (fp + tn)
        precision[c] = tp / (tp + fp)
        recall[c] = sensitivity[c]
    f1_score = 2 * precision * recall / (precision + recall)

    class_wise_metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
    }  # 'Recall': recall is same with sensitivity
    return class_wise_metrics


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            if s.size(0) % config["test_crop_multiple"] != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={config['test_crop_multiple']}."
                )

            real_minibatch = s.size(0) // config["test_crop_multiple"]
            s_ = torch.zeros((real_minibatch, s.size(1)))
            y_ = torch.zeros((real_minibatch,), dtype=torch.int32)

            for m in range(real_minibatch):
                s_[m] = s[config["test_crop_multiple"] * m : config["test_crop_multiple"] * (m + 1)].mean(
                    dim=0, keepdims=True
                )
                y_[m] = y[config["test_crop_multiple"] * m]

            s = s_
            y = y_

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            real_minibatch = sample_batched["signal"].size(0) // config["test_crop_multiple"]
            s_merge = torch.zeros((real_minibatch, config["out_dims"]))
            y_merge = torch.zeros((real_minibatch,), dtype=torch.int32)

            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            if s.size(0) % config["test_crop_multiple"] != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={config['test_crop_multiple']}."
                )

            for m in range(real_minibatch):
                s_merge[m] = s[config["test_crop_multiple"] * m : config["test_crop_multiple"] * (m + 1)].mean(
                    dim=0, keepdims=True
                )
                y_merge[m] = y[config["test_crop_multiple"] * m]

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            s = s_merge
            y = y_merge

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput
