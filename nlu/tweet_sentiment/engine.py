import torch
import sys
import math
import itertools
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from ..logger.logger import MetricLogger, SmoothedValue


def get_confusion_plot(predictions, targets, labels):
    cmatrix = confusion_matrix(y_true=targets, y_pred=predictions)

    df_cm = pd.DataFrame(cmatrix, index=labels, columns=labels)
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues')

    return fig


def train_one_epoch(model, optimizer, data_loader, weight, device, epoch, print_freq=10):
    model.train()
    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=['loss', 'lr'],
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})"
    )
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)

    weight = weight.to(device)

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        for k in batch.keys():
            batch[k] = batch[k].to(device)

        loss, _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            target=batch['target'],
            weight=weight)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(
            loss=loss_value, lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate_one_epoch(model, data_loader, weight, device, epoch, print_freq=10):
    model.eval()
    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=['loss'],
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})"
    )
    header = 'Eval:  [{}]'.format(epoch)

    weight = weight.to(device)

    predictions = []
    targets = []

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        for k in batch.keys():
            batch[k] = batch[k].to(device)

        loss, outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            target=batch['target'],
            weight=weight)

        loss_value = loss.item()

        metric_logger.update(loss=loss_value)

        predictions.append(outputs.detach().to(torch.device('cpu')).tolist())
        targets.append(batch['target'].detach().to(torch.device('cpu')).tolist())

    predictions = list(itertools.chain(*predictions))
    predictions = [np.argmax(el) for el in predictions]
    targets = list(itertools.chain(*targets))

    conf_matrix = get_confusion_plot(
        predictions, targets, list(model.class_map.keys()))

    report = classification_report(
        y_true=targets,
        y_pred=predictions,
        labels=list(model.class_map.values()),
        target_names=list(model.class_map.keys()),
        output_dict=True
    )

    return metric_logger, conf_matrix, report
