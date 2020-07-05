import torch
from typing import Dict
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
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


def get_df(class_map: Dict[str, int], data_path: str, n_el: int = None):
    df = pd.read_csv(data_path)
    df = df[df['text'].notnull()].reset_index(drop=True)
    df['sentiment_id'] = df['sentiment'].apply(lambda x: class_map[x])

    df['text_trf'] = df['text'] \
        .apply(lambda x: remove_common_misspellings(x.lower()))

    return df[:n_el]


class TweetSentimentDataset(Dataset):

    def __init__(self, data_path, tokenizer, n_el=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = 110  # based on training and test data analysis
        self.class_map = self.get_class_map()
        self.df = get_df(self.class_map, data_path, n_el)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        result = self.tokenizer.encode_plus(
            text=row['text_trf'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_max_length=True
        )

        result['idx'] = idx
        result['target'] = row['sentiment_id']

        return result

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def get_class_map():
        return {'negative': 0, 'neutral': 1, 'positive': 2}
