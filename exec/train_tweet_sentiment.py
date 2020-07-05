import os
import argparse
import torch
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from sklearn.model_selection import train_test_split

from nlu.tweet_sentiment.dataset import TweetSentimentDataset, collate_fn, get_df
from nlu.tweet_sentiment.models import BertClassification
from nlu.tweet_sentiment.engine import train_one_epoch, evaluate_one_epoch


def get_train_val_ds(data_path, pretrained_bert_tokenizer_path, size_tr_val, size_val):
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_tokenizer_path)

    ds_train_val = TweetSentimentDataset(data_path, tokenizer, n_el=size_tr_val)

    indices = torch.randperm(len(ds_train_val)).tolist()

    assert len(ds_train_val) > size_val, "validation set > data set"

    ds_train = Subset(ds_train_val, indices[:-size_val])
    ds_val = Subset(ds_train_val, indices[-size_val:])

    weights_train = None
    weights_val = None

    return ds_train, ds_val, weights_train, weights_val


def get_train_val_ds_stratified(data_path, pretrained_bert_tokenizer_path, size_tr_val, size_val):
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_tokenizer_path)

    df = get_df(
        class_map=TweetSentimentDataset.get_class_map(), data_path=data_path, n_el=size_tr_val)

    size_tr_val = len(df)
    assert size_tr_val > size_val, "validation set > data set"

    df_train, df_val, y_train, y_val = train_test_split(
        df, df['sentiment_id'], test_size=size_val / size_tr_val, stratify=df['sentiment_id'])

    weights_train = torch.tensor(
        (y_train.value_counts().min() / y_train.value_counts()).sort_index().astype('float32').values)
    weights_val = torch.tensor(
        (y_val.value_counts().min() / y_val.value_counts()).sort_index().astype('float32').values)

    ds_train = TweetSentimentDataset(data_path, tokenizer, n_el=size_tr_val)
    ds_val = TweetSentimentDataset(data_path, tokenizer, n_el=size_tr_val)

    ds_train.df = df_train  # all of the data is accessed through this dataframe
    ds_val.df = df_val

    return ds_train, ds_val, weights_train, weights_val


def write_summary(
        log_dir, epoch, model, logger_train=None, logger_val=None, report=None, conf_matrix=None
):
    with SummaryWriter(log_dir) as w:

        result_dict = defaultdict(dict)

        if logger_train is not None:
            for k, meter in logger_train.meters.items():
                result_dict[k].update({'train': meter.global_avg})

        if logger_val is not None:
            for k, meter in logger_val.meters.items():
                result_dict[k].update({'val': meter.global_avg})

        for k, v in result_dict.items():
            w.add_scalars(k, v, epoch)

        if report is not None:
            for k, v in report.items():
                if type(v) == dict:
                    if 'support' in v:
                        del v['support']
                    w.add_scalars(k, v, epoch)
                else:
                    w.add_scalar(k, v, epoch)

        if conf_matrix is not None:
            w.add_figure('confusion_matrix', conf_matrix, epoch)

        w.add_histogram('weights/linear/weight', model.linear.weight.data, epoch)
        w.add_histogram('weights/linear/bias', model.linear.bias.data, epoch)

        w.add_histogram('weights/batch_norm/weight', model.batch_norm.weight.data, epoch)
        w.add_histogram('weights/batch_norm/bias', model.batch_norm.bias.data, epoch)


def train(
        data_path: str,
        model_dir: str,
        log_dir: str,
        pretrained_bert_model_dir: str = 'bert-base-uncased',
        pretrained_bert_tokenizer_path: str = 'bert-base-uncased',
        size_tr_val: int = None,
        size_val: int = 300,
        batch_size: int = 4,
        num_epochs: int = 6,
        print_freq: int = 10,
        seed: int = 11,
):
    torch.manual_seed(seed)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # setup data set and data loader
    #
    ds_train, ds_val, weights_train, weights_val = get_train_val_ds_stratified(
        data_path, pretrained_bert_tokenizer_path, size_tr_val, size_val
    )

    dl_train = DataLoader(ds_train, batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False, collate_fn=collate_fn)

    # load model
    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epoch_range = range(1, num_epochs+1)

    model = BertClassification.from_pretrained(
        pretrained_bert_model_dir,
        output_hidden_states=False,
        output_attentions=False)

    model.to(device)

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.001)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # training
    #

    # evaluate the model with default weights
    logger_val, conf_matrix, report = evaluate_one_epoch(
        model, dl_val, weights_val, device, 0, print_freq)
    write_summary(log_dir, 0, model, None, logger_val, report, conf_matrix)

    for epoch in epoch_range:
        logger_train = train_one_epoch(
            model, optimizer, dl_train, weights_train, device, epoch, print_freq)

        lr_scheduler.step()

        logger_val, conf_matrix, report = evaluate_one_epoch(
            model, dl_val, weights_val, device, epoch, print_freq)

        torch.save(
            model.state_dict(),
            os.path.join(model_dir, f'model_state_dict_epoch_{epoch}.pth'))
        torch.save(
            optimizer.state_dict(),
            os.path.join(model_dir, f'optimizer_state_dict_epoch_{epoch}.pth'))

        write_summary(log_dir, epoch, model, logger_train, logger_val, report, conf_matrix)


if __name__ == "__main__":
    """
    Example: 
        python exec/train_tweet_sentiment.py \
            --data_path ${DATA_DIR}/train.csv \
            --model_dir ${RESULTS_DIR}/models \
            --log_dir ${RESULTS_DIR}/logs \
            --size_tr_val 10\
            --size_val 4\
            --batch_size 2 \
            --num_epochs 2 \
            --print_freq 2 \
            --seed 10
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, dest='data_path', help='data path')
    parser.add_argument('--model_dir', type=str, dest='model_dir',
                        help='Directory where trained models will be saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir',
                        help='Directory where training logs will be saved.')
    parser.add_argument('--pretrained_bert_model_dir', type=str, dest='pretrained_bert_model_dir',
                        default='bert-base-uncased',
                        help='Location of the pre-trained bert model (`state_dict` and `config`).\n'
                             'The default value triggers a download of these components from a s3-repo. \n')
    parser.add_argument('--pretrained_bert_tokenizer_path', type=str, dest='pretrained_bert_tokenizer_path',
                        default='bert-base-uncased',
                        help='Vocabulary path for the tokenizer.\n'
                             'The default value triggers a download of this component from a s3-repo.\n')
    parser.add_argument('--size_tr_val', type=int, dest='size_tr_val', default=None,
                        help='Total size of the train and validation dataset.\n'
                             'Used mainly to test the algorithm with a small amount of data.')
    parser.add_argument('--size_val', type=int, dest='size_val', default=300, help='Size of the validation set.')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=10,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, dest='print_freq', default=10,
                        help='Print training info every `print_freq` epochs.')
    parser.add_argument('--seed', type=int, dest='seed', default=11, help='seed')

    args = parser.parse_args()

    train(**vars(args))
