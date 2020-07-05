import sys
import math
import torch

from ..logger.logger import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=['loss', 'lr'],
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})"
    )
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        for k in batch.keys():
            batch[k] = batch[k].to(device)

        loss, loss_dict, _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            target=batch['target'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(
            loss=loss_value, lr=optimizer.param_groups[0]["lr"], **loss_dict)

    return metric_logger


@torch.no_grad()
def evaluate_one_epoch(model, data_loader, device, epoch, print_freq=10):
    model.eval()
    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=['loss'],
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})"
    )
    header = 'Eval:  [{}]'.format(epoch)

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        for k in batch.keys():
            batch[k] = batch[k].to(device)

        loss, loss_dict, _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            target=batch['target'])

        loss = loss.sum()
        loss_value = loss.item()

        metric_logger.update(loss=loss_value, **loss_dict)

    return metric_logger
