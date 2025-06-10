import os
import re
import ast
import json
import jieba
import pickle
import random
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from gensim import corpora, models


def ensure_dir(path: str) -> None:
    '''Make sure the directory exists
    
    Args: 
        path(str): the path of the directory
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def read_cascades(cascades_path: str) -> dict:
    """Read the cascades file
    
    Args:
        cascades_path(str): the path of the cascades file
    
    Returns:
        cascades(dict): the cascades of each post in dataset
    """
    cascades = defaultdict(list)
    
    split_char_1 = ' '
    split_char_2 = ','
    
    with open(cascades_path, 'r') as f:
        for line in f:
            chunks = line.strip().split(split_char_1)
            post = int(chunks[0])
            for chunk in chunks:
                if len(chunk.split(split_char_2)) == 2:
                    user, timestamp = chunk.split(split_char_2)
                    cascades[post].append((user, timestamp))
                    
    del_idx = []
    for post, cas in cascades.items():
        if len(cas) < 3:
            del_idx.append(post)
        elif len(cas) > 500:
            cascades[post] = cas[:500]
    for idx in del_idx:
        del cascades[idx]
    return cascades

def split_cascades(cascades: dict, output_folder: str) -> None:
    """Split the cascades to train, valid and test
    
    Args:
        cascades(dict): the cascades of each post id in dataset
        output_folder(str): the path of the output folder
    """
    with open(os.path.join(output_folder, 'cascade.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas[:-2]]) + '\n')
    with open(os.path.join(output_folder, 'cascadevalid.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas[:-1]]) + '\n')
    with open(os.path.join(output_folder, 'cascadetest.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas]) + '\n')
    
    # index the user
    users = set()
    for cas in cascades.values():
        users.update([user for user, _ in cas])
    user2id = {user: idx for idx, user in enumerate(users)}
    with open(os.path.join(output_folder, 'u2idx.pickle'), 'wb') as f:
        pickle.dump(user2id, f)
    id2user = [user for user, _ in sorted(user2id.items(), key=lambda x: x[1])]
    with open(os.path.join(output_folder, 'idx2u.pickle'), 'wb') as f:
        pickle.dump(id2user, f)
    print(f'User number: {len(user2id)}')  
        
def split_cascades4model(cascades: dict, output_folder: str) -> None:
    """Split the cascades to train, valid and test for ReChorus
    
    Args:
        cascades(dict): the cascades of each post id in dataset
        output_folder(str): the path of the output folder
    """
    train_df, dev_df, test_df = list(), list(), list()
    for post, cascade in cascades.items():
        for cas in cascade[:-2]:
            train_df.append({'item': cas[0], 'user': post, 'time': cas[1]})
        dev_df.append({'item': cascade[-2][0], 'user': post, 'time': cascade[-2][1]})
        test_df.append({'item': cascade[-1][0], 'user': post, 'time': cascade[-1][1]})
    
    train_df = pd.DataFrame(train_df)
    dev_df = pd.DataFrame(dev_df)
    test_df = pd.DataFrame(test_df)
    
    # index the user and post
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    post2id = {post: idx+1 for idx, post in enumerate(all_df['user'].unique())}
    user2id = {user: idx+1 for idx, user in enumerate(all_df['item'].unique())}
    for type, df in {'train': train_df, 'dev': dev_df, 'test': test_df}.items():
        df['user_id'] = df['user'].map(post2id)
        df['item_id'] = df['item'].map(user2id)
        df.drop(columns=['item', 'user'], inplace=True)
        df.sort_values(by=['time'], inplace=True)
        df.to_csv(os.path.join(output_folder, f'{type}.csv'), index=False, sep='\t')
    user_meta = pd.DataFrame({'item_id': list(user2id.values()), 'user_name': list(user2id.keys())})
    user_meta.to_csv(os.path.join(output_folder, f'item_meta.csv'), index=False, sep='\t')
    post_meta = pd.DataFrame({'user_id': list(post2id.values()), 'post_name': list(post2id.keys())})
    post_meta.to_csv(os.path.join(output_folder, f'user_meta.csv'), index=False, sep='\t')
    print(f'Total User number: {len(user2id)}', f'Total Post number: {len(post2id)}')

if __name__ == '__main__':
    dataset = 'Douban'
    
    cascades_path = f'./{dataset}/origin/cascades.txt'
    cascades = read_cascades(cascades_path)
    
    # split the cascades to train, valid and test dataset
    output_folder = f'./{dataset}/BaselineFormat/'
    ensure_dir(output_folder)
    split_cascades(cascades, output_folder) # edges is same as the social graph, so we can use the edges directly
    
    # split the cascades to train, valid and test dataset
    output_folder = f'./{dataset}/ModelFormat/'
    ensure_dir(output_folder)
    split_cascades4model(cascades, output_folder)
    
