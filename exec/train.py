import os
import argparse
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from nlu.models_pytorch.dataset import GoogleQADataset, collate_fn
from nlu.models_pytorch.models import BertMultipleProba
from nlu.models_pytorch.engine import train_one_epoch, evaluate_one_epoch


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
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_tokenizer_path)

    ds_train_val = GoogleQADataset(data_path, tokenizer, n_el=size_tr_val, max_q_length=254)

    indices = torch.randperm(len(ds_train_val)).tolist()

    assert len(ds_train_val) > size_val, "validation set > data set"

    ds_train = Subset(ds_train_val, indices[:-size_val])
    ds_val = Subset(ds_train_val, indices[-size_val:])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # load model
    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epoch_range = range(num_epochs)

    model = BertMultipleProba.from_pretrained(
        pretrained_bert_model_dir,
        output_hidden_states=False,
        output_attentions=False)

    model.to(device)

    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=0.01)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # (continue) training
    #
    for epoch in epoch_range:

        logger_train = train_one_epoch(model, optimizer, dl_train, device, epoch, print_freq)

        lr_scheduler.step(epoch=epoch + 1)  # set the lr for the next epoch? ###TODO:

        logger_val = evaluate_one_epoch(model, dl_val, device, epoch, print_freq)

        torch.save(model.state_dict(), os.path.join(model_dir, f'model_state_dict_epoch_{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, f'optimizer_state_dict_epoch_{epoch}.pth'))

        with SummaryWriter(log_dir) as w:
            for k, meter in logger_train.meters.items():
                w.add_scalars(k, {'train': meter.global_avg}, epoch)

            for k, meter in logger_val.meters.items():
                w.add_scalars(k, {'val': meter.global_avg}, epoch)


if __name__ == "__main__":
    """
    Example: 
        python exec/train.py \
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
