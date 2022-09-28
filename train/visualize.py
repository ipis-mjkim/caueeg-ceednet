from itertools import cycle
import warnings
import numpy as np
from sklearn.metrics import roc_curve, auc  # roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import wandb

from train.evaluate import calculate_class_wise_metrics


def draw_lr_search_record(learning_rate_record, use_wandb=False):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic

    fig = plt.figure(num=1, clear=True, constrained_layout=True, figsize=(7.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Learning Rate Search')
    ax.set_xlabel('Learning rate in log-scale')
    ax.set_ylabel('Accuracy')

    train_accs = np.array([[log_lr, tr] for log_lr, tr, vl in learning_rate_record])
    val_accs = np.array([[log_lr, vl] for log_lr, tr, vl in learning_rate_record])
    midpoints = np.array([[log_lr, (tr + vl)/2] for log_lr, tr, vl in learning_rate_record])

    ax.plot(train_accs[:, 0], train_accs[:, 1], 'o',
            color='tab:red', alpha=0.6, label='Train')
    ax.plot(val_accs[:, 0], val_accs[:, 1], 'o',
            color='tab:blue', alpha=0.6, label='Validation')
    ax.plot(midpoints[:, 0], midpoints[:, 1], '-',
            color='tab:purple', alpha=0.8, linewidth=1.0, label='Midpoint')

    midpoints = np.array([(tr + vl) / 2 for _, tr, vl in learning_rate_record])
    induces = np.argwhere(midpoints == np.max(midpoints))
    starting_log_lr = np.average(np.array([log_lr for log_lr, _, _ in learning_rate_record])[induces])

    ax.plot(starting_log_lr, np.max(midpoints), 'o',
            color='cyan', alpha=0.8, linewidth=2.0, label='Start LR')

    ax.legend(loc='lower center', fancybox=True, framealpha=0.7).get_frame().set_facecolor('white')

    if use_wandb:
        warnings.filterwarnings(action='ignore')
        wandb.log({"Learning Rate Search": mpl_to_plotly(fig)})
        warnings.filterwarnings(action='default')
    else:
        plt.show()

    fig.clear()
    plt.close(fig)


def draw_loss_plot(losses, lr_decay_step=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    fig = plt.figure(num=1, clear=True, figsize=(8.0, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    N = len(losses)
    x = np.arange(1, N + 1)
    ax.plot(x, losses)

    if lr_decay_step is None:
        pass
    elif type(lr_decay_step) is list:
        ax.vlines(lr_decay_step, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    else:
        x2 = np.arange(lr_decay_step, N, lr_decay_step)
        ax.vlines(x2, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    # ax.vlines([1, N], 0, 1, transform=ax.get_xaxis_transform(),
    #           colors='k', alpha=0.7, linestyle='solid')

    ax.set_xlim(left=0)
    ax.set_title('Loss Plot')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')

    plt.show()
    fig.clear()
    plt.close(fig)


def draw_accuracy_history(train_acc_history, val_acc_history, history_interval, lr_decay_step=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    fig = plt.figure(num=1, clear=True, figsize=(8.0, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    N = len(train_acc_history) * history_interval
    x = np.arange(history_interval, N + 1, history_interval)
    ax.plot(x, train_acc_history, 'r-', label='Train accuracy')
    ax.plot(x, val_acc_history, 'b-', label='Validation accuracy')

    if lr_decay_step is None:
        pass
    elif type(lr_decay_step) is list:
        ax.vlines(lr_decay_step, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    else:
        x2 = np.arange(lr_decay_step, N + 1, lr_decay_step)
        ax.vlines(x2, 0, 1, transform=ax.get_xaxis_transform(),
                  colors='m', alpha=0.5, linestyle='solid')
    # ax.vlines([history_interval, N], 0, 1, transform=ax.get_xaxis_transform(),
    #           colors='k', alpha=0.7, linestyle='solid')

    ax.set_xlim(left=0)
    ax.legend(loc='lower right')
    ax.set_title('Accuracy Plot during Training')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')

    plt.show()
    fig.clear()
    plt.close(fig)


def draw_heatmap(data, row_labels, col_labels, ax,
                 draw_cbar=False, cbar_label="", imshow_kw=None, cbar_kw=None):
    """ Draw a heatmap from a numpy array and two lists of labels.

    Args:
        data (np.array): A 2D numpy array of shape (M, N).
        row_labels (list): A list or array of length M with the labels for the rows.
        col_labels (list): A list or array of length N with the labels for the columns.
        ax (matplotlib.axes.Axes): A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
        draw_cbar (bool): Whether to draw the colormap or not.
        cbar_label (str, optional): The label for the colorbar.
        imshow_kw (dict): All other arguments are forwarded to `imshow`.
        cbar_kw (dict, optional): A dictionary with arguments to `matplotlib.Figure.colorbar`.

    Returns:
        im (AxesImage): The drawn AxesImage.
    """
    if imshow_kw is None:
        imshow_kw = dict()

    if cbar_kw is None:
        cbar_kw = dict()

    # Plot the heatmap
    im = ax.imshow(data, interpolation='nearest', **imshow_kw)

    # Create colorbar
    # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0 + 0.005, 0.02, ax.get_position().height])
    if draw_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, anno_format="{x:.2f}",
                     text_colors=("black", "white"),
                     threshold=None, text_kw=None):
    """ A function to annotate a heatmap.

    Args:
        im (AxesImage): The AxesImage to be labeled.
        data (numpy array, optional): Data to annotate. If None (the default) uses the data recorded in AxesImage.
        anno_format (str, optional): The format of the annotations inside the heatmap.
        text_colors (list of str, optional): A pair of colors.
            The first is used for values below a threshold, the second for those above.
        threshold (float, optional): Ratio in data units according to which the colors from text_colors are applied.
            If None (the default) uses the middle of the colormap as separation.
        text_kw (dict): All other arguments are forwarded to each call to `text` used to create the text labels.

    Returns:
        texts (list of str):
    """
    if text_kw is None:
        text_kw = dict()

    if data is None:
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is None:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by text_kw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              size="large")
    kw.update(text_kw)

    # Get the formatter in case a string is supplied
    if isinstance(anno_format, str):
        anno_format = matplotlib.ticker.StrMethodFormatter(anno_format)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=text_colors[int(im.norm(data[i, j]) > threshold)])
            im.axes.text(j, i, anno_format(data[i, j], None), **kw)


def draw_confusion(confusion, class_label_to_name, normalize=False, use_wandb=False, save_path=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    H = len(class_label_to_name) + 0.5
    W = len(class_label_to_name) + 0.5
    fig = plt.figure(num=1, clear=True, figsize=(W, H), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    data = confusion
    anno_format = "{x:d}"
    if normalize:
        data = confusion / confusion.sum(axis=1, keepdims=True)
        anno_format = "{x:.2f}"

    im = draw_heatmap(data, class_label_to_name, class_label_to_name,
                      ax=ax, imshow_kw={'alpha': 0.9, 'cmap': "YlOrRd"},  # jet, YlOrRd, RdPu
                      draw_cbar=False, cbar_label="", cbar_kw={})

    annotate_heatmap(im, anno_format=anno_format, text_colors=("black", "white"), threshold=0.7)

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')

    # save
    if save_path:
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({'font.family': 'Arial'})
        plt.rcParams["savefig.dpi"] = 1200
        fig.savefig(save_path, transparent=True)

    # draw
    if use_wandb:
        wandb.log({'Confusion Matrix (Image)': wandb.Image(plt)})

    if save_path is None and use_wandb is False:
        plt.show()

    # fig.clear()
    plt.close(fig)


def draw_class_wise_metrics(confusion, class_label_to_name, use_wandb=False, save_path=None):
    class_wise_metrics = calculate_class_wise_metrics(confusion)

    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    H = len(class_label_to_name) + 0.5
    W = len(class_wise_metrics) + 0.5
    fig = plt.figure(num=1, clear=True, figsize=(W, H), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    im = draw_heatmap(data=np.array([*class_wise_metrics.values()]).T,  # np.ones((C, len(class_wise_metrics))),
                      row_labels=class_label_to_name, col_labels=[*class_wise_metrics.keys()],
                      ax=ax, imshow_kw={'alpha': 0.9, 'cmap': "YlOrRd"},  # jet, YlOrRd, RdPu
                      draw_cbar=False,  cbar_label="", cbar_kw={'alpha': 0.9})

    annotate_heatmap(im, data=np.array([*class_wise_metrics.values()]).T,
                     anno_format="{x:.2f}", text_colors=("black", "white"), threshold=0.7)

    ax.set_title('Class-wise metrics')

    # save
    if save_path:
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({'font.family': 'Arial'})
        plt.rcParams["savefig.dpi"] = 1200
        fig.savefig(save_path, transparent=True)

    # draw
    if use_wandb:
        wandb.log({'Class-wise Metrics (Image)': wandb.Image(plt)})

    if save_path is None and use_wandb is False:
        plt.show()

    # fig.clear()
    plt.close(fig)


def draw_roc_curve(score, target, class_label_to_name, use_wandb=False, save_path=None):
    plt.style.use('default')  # default, ggplot, fivethirtyeight, classic
    lw = 1.1

    # Binarize the output
    n_classes = len(class_label_to_name)
    target = label_binarize(target, classes=np.arange(n_classes))

    if n_classes == 2 and target.shape[1] == 1:
        target_temp = np.zeros((target.shape[0], 2), dtype=target.dtype)
        target_temp[:, [0]] = (target == 0)
        target_temp[:, [1]] = target
        target = target_temp

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally, average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # draw class-agnostic ROC curve
    fig = plt.figure(num=1, clear=True, figsize=(8.5, 4.0), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    colors = cycle(['limegreen', 'mediumpurple', 'darkorange',
                    'dodgerblue', 'lightcoral', 'goldenrod',
                    'indigo', 'darkgreen', 'navy', 'brown'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='{0} (area = {1:0.2f})'
                      ''.format(class_label_to_name[i], roc_auc[i]))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Class-Wise ROC Curves')
    ax.legend(loc="lower right")

    # Plot class-aware ROC curves
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='-', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle='-', linewidth=lw)

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Class-Agnostic ROC Curves')
    ax.legend(loc="lower right")

    # save
    if save_path:
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({'font.family': 'Arial'})
        plt.rcParams["savefig.dpi"] = 1200
        fig.savefig(save_path, transparent=True)

    # draw
    if use_wandb:
        wandb.log({'ROC Curve (Image)': wandb.Image(plt)})

    if save_path is None and use_wandb is False:
        plt.show()

    # fig.clear()
    plt.close(fig)
