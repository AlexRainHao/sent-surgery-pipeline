"""pass"""

from typing import List, Text
from collections import defaultdict
from itertools import chain

class Vocab:
    """pass"""
    
    def __init__(self):
        self.tk2idx = defaultdict(int)
        self.idx2tk = defaultdict(str)
        self.group_idx = defaultdict(list)
        self.tk2score = defaultdict(int)
    
    @property
    def postive_name(self):
        return "postive"
    
    @property
    def negtive_name(self):
        return "negtive"
    
    @property
    def alters_name(self):
        return "alters"
    
    @classmethod
    def gene_from_list(cls, tks: List[Text], name = None, score = 0):
        vocab = Vocab()
        vocab.add_seq(tks, name = name, score = score)
        return vocab
    
    def initialize(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def add(self, token, score=1):
        if token not in self.tk2idx:
            idx = len(self.tk2idx)
            self.tk2idx[token] = idx
            self.idx2tk[idx] = token
            self.tk2score[token] = score
    
    def add_seq(self, seq: List[Text], name=None, score=1):
        
        if name:
            _start = len(self.tk2idx)
            _end = _start + len(set(seq))
            self.group_idx[name] += list(range(_start, _end))
        
        [self.add(tk, score=score) for tk in seq]
    
    def init_tks(self, tokens: List[Text]):
        [self.add(tk) for tk in tokens]
    
    def istk(self, tk):
        return tk in self.tk2idx
    
    def get_tk(self, idx):
        return self.idx2tk[idx]
    
    def get_tks(self, idx: List[int]):
        return [self.get_tk(id) for id in idx]
    
    def get_group(self, name):
        """pass"""
        return self.group_idx.get(name, [])
    
    def get_all_group(self):
        seed_idx = self.get_group(self.alters_name)
        pos_idx = self.get_group(self.postive_name)
        neg_idx = self.get_group(self.negtive_name)
        return seed_idx, pos_idx, neg_idx
    
    def get_group_tks(self, name):
        idx = self.get_group(name)
        return self.get_tks(idx)
    
    def get_emo_group_tks(self):
        return self.get_group_tks(self.postive_name) + self.get_group_tks(self.negtive_name)
    
    def get_score(self, tk):
        return self.tk2score.get(tk, 0)
    
    def size(self):
        return self.__len__()
    
    def reverse_group_idx(self):
        """from group->idx to idx->group"""
        idx2group = defaultdict(str)
        for group, indexes in self.group_idx.items():
            for idx in indexes:
                idx2group[idx] = group
        return idx2group
        
    def __len__(self):
        return len(self.tk2idx)
    
    def __add__(self, vob):
        """merge another vocab,
        only return a new vocab that merging pair of vocab,
        not change the original vocabs
        """
        b_tk2idx, b_idx2tk, b_group_idx, b_tk2score = self._redirect_index(force_start = 0, out = False)
        _force_step = len(b_tk2idx) + 1
        t_tk2idx, t_idx2tk, t_group_idx, t_tk2score = vob._redirect_index(force_start = _force_step, out = False)
        
        b_tk2idx.update(t_tk2idx)
        b_idx2tk = {idx: tk for tk, idx in b_tk2idx.items()}

        for _gp, _val in t_group_idx.items():
            b_group_idx[_gp] += _val

        b_tk2score.update(t_tk2score)

        res_vocab = Vocab()
        res_vocab.initialize(tk2idx = b_tk2idx,
                             idx2tk = b_idx2tk,
                             group_idx = b_group_idx,
                             tk2score = b_tk2score)
        res_vocab._redirect_index(force_start = 0, out = True)
        
        return res_vocab
    
    def merge(self, vocab):
        """pass"""
        self.__add__(vocab)
    
    def _redirect_index(self, force_start = 0, out = True):
        tk2idx = defaultdict(int)
        idx2tk = defaultdict(str)
        group_idx = defaultdict(list)
        idx_group = self.reverse_group_idx()
        
        for idx, tk in self.idx2tk.items():
            tk2idx[tk] = force_start
            idx2tk[force_start] = tk
            group_idx[idx_group[idx]].append(force_start)
            
            force_start += 1
        
        if out:
            self.tk2idx = tk2idx
            self.idx2tk = idx2tk
            self.group_idx = group_idx
        
        return tk2idx, idx2tk, group_idx, self.tk2score
        
    
class BaseVocab(Vocab):
    # TODO: base emo, extreme, deny vocab
    pass
