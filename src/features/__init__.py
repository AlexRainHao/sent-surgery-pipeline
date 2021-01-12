"""pass"""

from typing import List, Text, Dict, Any, Union
import numpy as np

from src.vocab import Vocab

def doc_onehot_mat(doc_tokens: List[List[Text]],
                   vocab: Vocab):
    """pass"""
    
    tk2idx = vocab.tk2idx
    all_tks = list(tk2idx.keys())

    onehot_mat = np.zeros(shape=(vocab.size() + 1, len(doc_tokens)), dtype=np.int8)

    for id, doc in enumerate(doc_tokens):
        tks = list(map(lambda tk: tk2idx[tk] if tk in doc else -1, all_tks))
        onehot_mat[tks, id] = 1

    return onehot_mat


def doc_label_mat(doc_labels: List[int]):
    """only support Binary Classifier
    TODO: support multi-label
    """
    label_mat = np.asarray(doc_labels, dtype = np.int8)
    label_mat = np.where(label_mat > 0, label_mat, -1)
    return label_mat


def token_emotion_mat(vocab: Vocab):
    """pass"""
    emotion_mat = np.zeros(shape=(vocab.size()))
    emotion_mat[vocab.get_group(vocab.postive_name)] = 1
    emotion_mat[vocab.get_group(vocab.negtive_name)] = -1

    return emotion_mat


def ngrams(tokens: List[Text], window: int) -> List:
    """pass"""
    assert window > 0, "window must greater than 0"
    
    window = min(window, len(tokens))
    
    return list(zip(*[tokens[i:] for i in range(window)]))