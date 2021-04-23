import torch as th
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import json


class UCCDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        """Expects data of type dataframe with columns "text" and "labels" """
        self.tokenizer = tokenizer
        self.text = data.text
        self.labels = data.labels  # data.label 
        self.max_len = max_len
        self.n_datapoints = len(self.text)

    def __len__(self):
        return self.n_datapoints

    def __getitem__(self, idx):
        text = str(self.text[idx])
        inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': th.flatten(inputs['input_ids']).type(th.long),
            'token_type_ids': th.flatten(
                inputs['token_type_ids']).type(th.long),
            'attention_mask': th.flatten(
                inputs['attention_mask']).type(th.long),
            'labels': th.tensor(self.labels[idx], dtype=th.long)  # dtype=th.float for crossentropyloss?
        }


def remove_zero_labels(df):
    new_df = pd.DataFrame(columns=df.columns)
    for idx, row in df.iterrows():
        if not (row['healthy'] == 0 and row['unhealthy'] == 0):
            new_df = new_df.append(row)
    return new_df



def dataframe_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    train_df = pd.DataFrame(data=data)
    return train_df


def make_binary_df(clean_df, label_idx):
    """
       Expects a clean df from clean_df.

       label_idx is the idx for the relevant characteristic/label in df.
       Comments where the given characteristic is present will be given
       the label 1, other comments will have label 0

       label_idx mapping:
       0 = healthy
       1 = sarcastic
       2 = generalisation_unfair
       3 = hostile
       4 = antagonise
       5 = condescending
       6 = dismissive
       7 = unhealthy, special handling
    """
    df = clean_df
    if label_idx != 7:    
        for idx, row in df.iterrows():
            if row.labels[label_idx] == 1:  # if it has the characteristic
                new_label = 1
            else:
                new_label = 0
            row.labels = new_label
        return df
    else:
        for idx, row in df.iterrows():
            if row.labels[0] == 0:  # if the comment is not healthy, it is considered unhealthy
                new_label = 1
            else:
                new_label = 0
            row.labels = new_label
        return df


def oversampling(df):
    max_size = df['labels'].value_counts().max()
    new_df = [df]
    for idx, group in df.groupby('labels'):
        new_df.append(group.sample(max_size-len(group), replace=True))
    df = pd.concat(new_df)
    return df


def balance_df(binary_df):
    """Expects binary dataframe from make_binary_df"""
    balanced_df = pd.DataFrame()
    negative_samples = 0
    max_neg = sum(binary_df.labels)
    for idx, row in binary_df.iterrows():
        if row.labels == 0 and negative_samples < max_neg:
            balanced_df = balanced_df.append(row)
            negative_samples += 1
        elif row.labels == 1: #  has characeristic
            balanced_df = balanced_df.append(row)
    return balanced_df


def update_dataset(row, dataset):
    dataset['text'].append(row['text'])
    dataset['labels'].append(row['labels'])
    return dataset


def balance_and_write_dataset(csv_file):
    raw_df = pd.read_csv(csv_file)
    clean_df = get_clean_df(raw_df)
    n_healthy = sum(raw_df['healthy'])
    n_unhealthy = len(raw_df) - n_healthy
    target_arr = np.array(clean_df['labels'].to_list())  # Nx7
    pure_healthy = 0
    for arr in target_arr:
        if np.array_equal(arr, np.array([0, 0, 0, 0, 0, 0, 1])):
            pure_healthy += 1
    n_unpure_healthy = n_healthy - pure_healthy
    # print(len(raw_df))
    # print(n_healthy)
    # print(n_unhealthy)
    # print(pure_healthy)
    # print(n_unpure_healthy)
    n_desired_pure_healthy = n_unhealthy - n_unpure_healthy
    # want equally many healthy as unhealthy
    # can try both mix of "nuanced" healthy and pure healthy,
    # and also only pure healthy

    # mix
    # loop through dataset and add all unhealthy, and nuanced healthy,
    # and pure healthy up to
    # limit of n_unhealthy - n_unpure_healthy
    balanced_dataset = {'text': list(), 'labels': list()}
    pure_healthy_counter = 0
    unpure_healthy_counter = 0
    unhealthy_counter = 0
    for idx, row in clean_df.iterrows():
        print(f'{idx}/{len(clean_df)}')
        label = np.array(row['labels'])
        healthy = bool(label[-1] == 1)
        if healthy:
            if (sum(label) == 1 and
                    pure_healthy_counter < n_desired_pure_healthy):
                pure_healthy_counter += 1
                balanced_datset = update_dataset(row, balanced_dataset)
            elif sum(label) != 1:
                balanced_datset = update_dataset(row, balanced_dataset)
                unpure_healthy_counter += 1
        else:
            balanced_datset = update_dataset(row, balanced_dataset)
            unhealthy_counter += 1
    print(len(balanced_dataset))
    print(pure_healthy_counter)
    print(unpure_healthy_counter)
    print(unhealthy_counter)
    with open('data/train_balanced.json', 'w') as f:
        json.dump(balanced_dataset, f)


def add_unhealthy_columns(df):
    healthy = list(df.healthy)
    unhealthy = [int(not x) for x in healthy]
    conf = list(df['healthy:confidence'])
    df['unhealthy'] = unhealthy
    df['unhealthy:confidence'] = conf
    return df


def get_clean_df(raw_df):
    new_df = pd.DataFrame(columns=['text', 'labels'])
    new_df['labels'] = raw_df[
        ['healthy',
         'sarcastic',
         'generalisation_unfair',
         'hostile',
         'antagonise',
         'condescending',
         'dismissive'
         ]].values.tolist()
    new_df['text'] = raw_df['comment']
    return new_df


def get_highest_confidence_datapoints(df, characteristic, n_points='all', pure_only=False):
    df = df.loc[df[characteristic] == 1]
    if pure_only:
        df = df.loc[df[f'{characteristic}:confidence'] == 1]
        if n_points == 'all':
            return df.reset_index(drop=True)
        else:
            assert type(n_points) is int
            return df[:n_points].reset_index(drop=True)
    sorted_df = df.sort_values(f'{characteristic}:confidence', ascending=False)
    assert len(sorted_df) >= n_points
    return sorted_df[:n_points].reset_index(drop=True)


def english_oversampling(nor_df, eng_df, characteristic, n_eng='all', pure_only=False):
    """Takes in raw dfs and returns clean dfs"""
    eng_df = add_unhealthy_columns(eng_df)
    high_conf = get_highest_confidence_datapoints(
        eng_df,
        characteristic,
        n_eng,
        pure_only=pure_only
    )
    clean_high_conf = get_clean_df(high_conf)
    clean_nor = get_clean_df(nor_df)
    df = pd.concat([clean_high_conf, clean_nor])
    return df.sample(frac=1).reset_index(drop=True)


if __name__ == '__main__':
    eng_df = pd.read_csv('data/UCC/train.csv')
    nor_df = pd.read_csv('data/norwegian/val.csv')
    print(len(nor_df))
    eng_oversampled = english_oversampling(nor_df, eng_df, 'unhealthy', pure_only=True)
    print(len(eng_oversampled))
    binary_eng_oversampled = make_binary_df(eng_oversampled, 7)
    binary_df = binary_eng_oversampled.sample(frac=1).reset_index(drop=True)
    # print(binary_df)