import torch
from collections import defaultdict
from typing import Dict, List
from torch.utils.data import Dataset
import pandas as pd
from .variables import target_cols
from ..clean_txt import remove_common_misspellings


def collate_fn(batch):
    res = defaultdict(list)

    for el in batch:
        for k, v in el.items():
            res[k].append(v)

    res = dict(res)

    for k, v in res.items():
        res[k] = torch.tensor(v)

    return res


def get_df(data_path: str, n_el: int = None):
    df = pd.read_csv(data_path, nrows=n_el)

    df['answer_trf'] = df['answer'] \
        .apply(lambda x: remove_common_misspellings(x).lower())

    df['question_trf'] = df[['question_title', 'question_body']] \
        .apply(lambda x: remove_common_misspellings(x[0].lower()) + ' [SEP] ' +
                         remove_common_misspellings(x[1].lower()), 1)

    return df


def get_qa_tokens_and_token_types(q: str, a: str, tokenizer, max_q_length=254) \
        -> Dict[str, List[int]]:
    max_len = tokenizer.max_len
    assert max_q_length + 3 <= max_len, 'Decrease the  maximum possible question length'

    q = tokenizer.encode(
        text=q, add_special_tokens=False, truncation=True, max_length=max_q_length)
    a = tokenizer.encode(
        text=a, add_special_tokens=False, truncation=True, max_length=max_len - len(q) - 3)

    res = tokenizer.encode_plus(
        text=q,
        text_pair=a,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        pad_to_max_length=True,
        # return_tensors='pt',
    )

    return res


class GoogleQADataset(Dataset):

    def __init__(self, data_path, tokenizer, n_el=None, max_q_length=254):
        self.data_path = data_path
        self.target_cols = target_cols
        self.tokenizer = tokenizer
        self.max_q_length = max_q_length
        self.df = get_df(data_path, n_el)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        result = get_qa_tokens_and_token_types(
            q=row['question_trf'],
            a=row['answer_trf'],
            tokenizer=self.tokenizer,
            max_q_length=self.max_q_length)

        result['qa_id'] = row['qa_id'].astype('int64')

        if all(x in row.keys() for x in self.target_cols):
            result['target'] = row[self.target_cols].values.astype('float32')

        return result

    def __len__(self):
        return self.df.shape[0]
