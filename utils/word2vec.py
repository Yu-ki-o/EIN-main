import os
import os.path as osp
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from utils.tools import word_tokenizer


class Embedding():
    def __init__(self, w2v_path, lang, tokenize_mode):
        self.w2v_path = w2v_path
        self.lang = lang
        self.tokenize_mode = tokenize_mode
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = self.make_embedding()

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        self.embedding_matrix = []
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
        for i, word in enumerate(self.embedding.wv.key_to_index):
            # e.g. self.word2index['魯'] = 1
            # e.g. self.index2word[1] = '魯'
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv.get_vector(word, norm=True))
        self.embedding_matrix = torch.from_numpy(
            np.asarray(self.embedding_matrix, dtype='float32')
        )
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def sentence_word2idx(self, sen):
        sentence_idx = []
        for word in word_tokenizer(sen, self.lang, self.tokenize_mode):
            if (word in self.word2idx.keys()):
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return sentence_idx

    def get_word_embedding(self, sen):
        sentence_idx = self.sentence_word2idx(sen)
        word_embedding = self.embedding_matrix[sentence_idx]
        return word_embedding

    def get_sentence_embedding(self, sen):
        word_embedding = self.get_word_embedding(sen)
        sen_embedding = torch.sum(word_embedding, dim=0)
        return sen_embedding

    def get_sentence_embeddings(self, sentences):
        return torch.stack([self.get_sentence_embedding(sentence) for sentence in sentences], dim=0)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


def collect_sentences(label_source_path, lang, tokenize_mode):
    sentences = collect_label_sentences(label_source_path)
    sentences = [word_tokenizer(sentence, lang=lang, mode=tokenize_mode) for sentence in sentences]
    return sentences


def collect_label_sentences(path):
    sentences = []
    for filename in os.listdir(path):
        filepath = osp.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(commnet['content'])
    return sentences


def collect_unlabel_sentences(path, unsup_train_size):
    sentences = []
    filenames = os.listdir(path)
    random.shuffle(filenames)
    for i, filename in enumerate(filenames):
        if i == unsup_train_size:
            break
        filepath = osp.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(commnet['content'])
    return sentences


def train_word2vec(sentences, vector_size, seed):
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=5, epochs=30, sg=1, seed=seed)
    return model


class MultilingualE5Embedding():
    def __init__(self, model_name='intfloat/multilingual-e5-base', device='cpu',
                 max_length=128, batch_size=64, local_files_only=False):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        print('Loading text encoder tokenizer: {} (local_files_only={})'.format(
            model_name, local_files_only
        ), flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )
        print('Loading text encoder model: {} -> {}'.format(
            model_name, self.device
        ), flush=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print('Text encoder loaded. Embedding dim: {}'.format(self.embedding_dim), flush=True)

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_sentence_embeddings(self, sentences):
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch_sentences = sentences[start:start + self.batch_size]
                batch_sentences = ['passage: {}'.format(sentence) for sentence in batch_sentences]
                batch_dict = self.tokenizer(
                    batch_sentences,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
                outputs = self.model(**batch_dict)
                batch_embeddings = self.average_pool(
                    outputs.last_hidden_state,
                    batch_dict['attention_mask']
                )
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

    def get_sentence_embedding(self, sentence):
        return self.get_sentence_embeddings([sentence]).squeeze(0)
