import os
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from nlu.models_pytorch.models import BertMultipleProba
from nlu.models_pytorch.dataset import GoogleQADataset, collate_fn


def predict(
        data_path: str,
        result_dir: str,
        model_dir: str,
        pretrained_bert_model_dir: str = 'bert-base-uncased',
        pretrained_bert_tokenizer_path: str = 'bert-base-uncased',
        load_epoch: int = 1,
        batch_size: int = 4,
        n_el: int = None
):
    os.makedirs(result_dir, exist_ok=True)

    # setup data set and data loader
    #
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_tokenizer_path)

    ds_test = GoogleQADataset(data_path, tokenizer, n_el=n_el, max_q_length=254)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # load model
    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertMultipleProba.from_pretrained(
        pretrained_bert_model_dir,
        output_hidden_states=False,
        output_attentions=False
    )
    model.load_state_dict(torch.load(
        os.path.join(model_dir, f'model_state_dict_epoch_{load_epoch}.pth'),
        map_location=device)
    )
    model.to(device)
    model.eval()

    # Prediction
    #
    outputs = []
    qa_ids = []

    for batch in dl_test:

        for k in batch.keys():
            batch[k] = batch[k].to(device)

        output = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'])

        outputs.append(output.detach().to(torch.device('cpu')).numpy())
        qa_ids.append(batch['qa_id'].detach().to(torch.device('cpu')).numpy())

    outputs = np.vstack(outputs)
    qa_ids = np.concatenate(qa_ids)

    results = pd.DataFrame(
        data=outputs,
        index=pd.Index(qa_ids, name='qa_id'),
        columns=ds_test.target_cols)

    result_path = os.path.join(result_dir, 'predictions.csv')
    results.to_csv(result_path)


if __name__ == "__main__":
    """ 
    Example:         
        python exec/predict.py \
            --data_path ${DATA_DIR}/test.csv \
            --result_dir ${RESULTS_DIR}/results \
            --model_dir ${RESULTS_DIR}/models \
            --load_epoch 1 \
            --batch_size 2 \
            --n_el 10
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, dest='data_path', help='data path')
    parser.add_argument('--result_dir', type=str, dest='result_dir',
                        help='Directory where model predictions will be saved.')
    parser.add_argument('--model_dir', type=str, dest='model_dir',
                        help='Directory where the model that will be loaded is saved.')
    parser.add_argument('--pretrained_bert_model_dir', type=str, dest='pretrained_bert_model_dir',
                        default='bert-base-uncased',
                        help='Location of the pre-trained bert model (`state_dict` and `config`).\n'
                             'The default value triggers a download of these components from a s3-repo. \n')
    parser.add_argument('--pretrained_bert_tokenizer_path', type=str, dest='pretrained_bert_tokenizer_path',
                        default='bert-base-uncased',
                        help='Vocabulary path for the tokenizer.\n'
                             'The default value triggers a download of this component from a s3-repo.\n')
    parser.add_argument('--load_epoch', type=int, dest='load_epoch',
                        help='The epoch index of the model that will be loaded')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=4,
                        help='Batch size. A parameter of the DataLoader')
    parser.add_argument('--n_el', type=int, dest='n_el', default=None,
                        help='Numbers of elements that will be processed.\n'
                             'Used only to test the algorithm with a small amount of data.')

    args = parser.parse_args()

    predict(**vars(args))
