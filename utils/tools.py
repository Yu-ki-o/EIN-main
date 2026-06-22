import torch
import numpy as np
import random
import importlib
import json
import shutil
import jieba
import nltk
from nltk.tokenize import MWETokenizer
import os
import re
from torch_geometric import seed_everything


mwe_tokenizer = MWETokenizer([('<', '@', 'user', '>'), ('<', 'url', '>')], separator='')

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text

def init_seed(seed, need_deepfix):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seed_everything(seed)
    if need_deepfix == True:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

def word_tokenizer(sentence, lang='en', mode='naive'):
    if lang == 'en':
        if mode == 'nltk':
            return mwe_tokenizer.tokenize(nltk.word_tokenize(sentence))
        elif mode == 'naive':
            return sentence.split()
    if lang == 'ch':
        if mode == 'jieba':
            return jieba.lcut(sentence)
        elif mode == 'naive':
            return sentence


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path, ignore_errors=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'), exist_ok=True)

    return train_path, val_path, test_path
