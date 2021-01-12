"""
Filter token for dataset using
    * token entropy
    * Info-Gain

This alg calculate each token's info gain once itself added to dataset,
then sorted all scores reversely that used to truncate top K token,
which as important as alg expected

the token list for each example feed into will be update by filtering results,
that all the non-significant token would be dropped

We also support a set of tokens that needed to consideration forced

E.X.
    >>> text = [list("abcdef"), list("bcbd"), list("abef"), list("feai"), list("abddi"), list("bcj")]
    >>> label = [1,-1,1,1,1,-1]
    >>> examples = [Example(''.join(exam), lb) for exam, lb in zip(text, label)]
    >>> for exam in examples:
    >>>     exam.set("tokens", list(exam.text))
    >>> op = tokenEntropyFilter()
    >>> op.add_force_tks(["j"])
    >>> tks = op.filter(examples, prop = 0.5)
    >>> print(tks)
        0.500% tokens filtered with entropy alg
        ['a', 'e', 'c', 'f']
    >>> op.run(examples, prop = 0.5)
    >>> print([exam.get("tokens") for exam in examples])
        0.500% tokens filtered with entropy alg
        0.000% examples have empty tokens
        [['a', 'c', 'e', 'f'], ['c'], ['a', 'e', 'f'], ['f', 'e', 'a'], ['a'], ['c', 'j']]
"""

from typing import List, Text, Dict, Any, Union
from itertools import chain
import numpy as np

from src.utils import Example
from src.vocab import Vocab
from src.features import doc_onehot_mat, doc_label_mat

class tokenEntropyFilter:
    
    name = "token_entropy_filter"

    def __init__(self,
                 force_tks: Union[List[Text], Vocab] = None):
        self.force_tks = self._load_force_tks(force_tks) # Vocab

    def run(self,
            examples: List[Example],
            prop: float = 1.,
            top_k: int = None,
            *args, **kwargs):
        """will change token property for each exam,
        and with a certain probability may make tokens empty
        """
        _args = locals().copy()
        _args.pop("self")
        
        remain_tks = self.filter(**_args)
        self.force_tks.add_seq(remain_tks)
        force_tks = self.force_tks.tk2idx
        
        zero_count = 0
        for idx, exam in enumerate(examples):
            _sub_remain_tks = list(filter(lambda tk: tk in force_tks, exam.get("tokens")))
            if len(_sub_remain_tks) == 0:
                zero_count += 1
    
            exam.set("tokens", _sub_remain_tks)

        # all example tokens be empty warning
        if zero_count == len(examples):
            print(f"Warnings, all examples have emtpy tokens")
            
        print(f"{zero_count / len(examples):.3f}% examples have empty tokens")
        

    def add_force_tks(self, tks: Union[Text, List[Text]]):
        """pass"""
        if isinstance(tks, Text):
            tks = [tks]
        
        self.force_tks.add_seq(tks)

    def filter(self, examples: List[Example],
               prop: float = 1.,
               top_k: int = None,
               *args, **kwargs) -> List[Text]:
        """pass"""
        doc_mat, vocab = self.create_onehot_mat([exam.get("tokens") for exam in examples])
        doc_label = doc_label_mat([exam.label for exam in examples])

        doc_entropy = self._doc_entropy(doc_label)
        tks_entropy = self._tks_entropy(doc_label, doc_mat)
    
        # entropy gain
        tks_entropy = doc_entropy - tks_entropy
        tks_size = tks_entropy.shape[0]
    
        # truncation if needed
        trun_count = self._truncate_size(prop, top_k, tks_size)
        remain_tks = self._truncate(tks_entropy, trun_count, vocab)
        
        if len(remain_tks) == 0:
            print("Warning, all examples tokens filtered")
        
        print(f"{1 - trun_count/tks_size:.3f}% tokens filtered with entropy alg")
        
        return remain_tks
    
    def create_onehot_mat(self, doc_tokens: List[List[Text]]) -> np.array:
        """pass"""
        vocab = Vocab()
        tks = list(set(chain.from_iterable(doc_tokens)))
        vocab.add_seq(tks)
        
        doc_mat = doc_onehot_mat(doc_tokens, vocab)
        
        return doc_mat[:-1], vocab
    
    def _load_force_tks(self, force_tks: Union[List, Vocab]):
        if force_tks is None:
            return Vocab()
        elif isinstance(force_tks, List):
            vocab = Vocab()
            vocab.add_seq(force_tks)
            return vocab
        else:
            return force_tks
    
    def _doc_entropy(self, doc_label: Union[np.array, List]):
        """pass"""
        return self._entropy(doc_label)
    
    def _tks_entropy(self, doc_label: np.array, doc_mat: np.array):
        """pass"""
        assert len(doc_mat.shape) == 2, "`doc_mat` must have shape of 2"
        
        doc_size = doc_label.squeeze().shape[0]

        def _self_func(row_line):
            # tk shows
            p_score, p_weight = self._tk_entropy(row_line, doc_label)
            # tk not shows
            n_score, n_weight = self._tk_entropy(1 - row_line, doc_label)
            return ((p_score * p_weight + n_score * n_weight) / doc_size)

        tks_entropy = np.apply_along_axis(_self_func, 1, doc_mat)
        
        return np.asarray(tks_entropy)
        
    
    def _entropy(self, labels: Union[np.array, List]):
        """pass"""
        labels = np.asarray(labels).flatten()
        if labels.size == 0:
            return 0

        label, count = np.unique(labels, return_counts = True)
        count_sum = count.sum()

        count = count / count_sum
        
        return -1. * np.sum(count * np.log(count))
    
    def _tk_entropy(self,
                    tk_label: Union[np.array, List],
                    doc_label: Union[np.array, List]):
        """pass"""
        tk_label = np.asarray(tk_label).squeeze() # [1,1,0,0,1]
        doc_label = np.asarray(doc_label) # [1,-1,1,-1,1]

        # filter zero count
        nzero_idx = np.where(tk_label != 0)

        pair_tk_doc = tk_label[nzero_idx] * doc_label[nzero_idx]
        return self._entropy(pair_tk_doc), len(nzero_idx[0])
    
    @staticmethod
    def _truncate_size(prop, top_k, tks_size):
        prop = 1. if prop < 0 else min(1., max(0., prop))
        top_k = max(1, top_k) if top_k else tks_size
        trun_count = int(min(prop * tks_size, top_k))
        
        return trun_count
    
    @staticmethod
    def _truncate(val: np.array, trun_count: int, vocab: Vocab = None):
        sort_index = np.argsort(val, axis = None)[::-1][:trun_count]
        if vocab:
            return vocab.get_tks(sort_index)
        else:
            return sort_index
    