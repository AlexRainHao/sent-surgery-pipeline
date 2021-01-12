"""
Filter token for dataset using
    * chi-square test

This alg calculate each token's chi-square score and sorted reversely,
top K tokens as the most-import tokens for corresponding dataset

E.X
    >>> text = [list("abcdef"), list("bcbd"), list("abef"), list("feai"), list("abddi"), list("bcj")]
    >>> label = [1, -1, 1, 1, 1, -1]
    
    >>> examples = [Example(''.join(exam), lb) for exam, lb in zip(text, label)]
    >>> for exam in examples:
    >>>     exam.set("tokens", list(exam.text))
    
    >>> op = tokenChiSquareFilter()
    >>> op.add_force_tks(["c"])
    >>> op.run(examples, prop = 0.5)
    >>> print([exam.get("tokens") for exam in examples])
        0.500% tokens filtered with entropy alg
        0.000% examples have empty tokens
        [['a', 'b', 'c'], ['b', 'c', 'b'], ['a', 'b'], ['a', 'i'], ['a', 'b', 'i'], ['b', 'c', 'j']]
"""

from typing import List, Text, Dict, Any, Union
import numpy as np

from src.utils import Example
from src.vocab import Vocab
from src.features import doc_label_mat
from src.features.token_entropy import tokenEntropyFilter

class tokenChiSquareFilter(tokenEntropyFilter):
    
    name = "token_chi_square_filter"
    
    def __init__(self, force_tks: Union[List[Text], Vocab] = None):
        super().__init__(force_tks)
    
    def filter(self, examples: List[Example],
               prop: float = 1.,
               top_k: int = None,
               *args, **kwargs) -> List[Text]:
        """pass"""
        doc_mat, vocab = self.create_onehot_mat([exam.get("tokens") for exam in examples])
        doc_label = doc_label_mat([exam.label for exam in examples])
        
        tks_chi = self._tks_chi(doc_label, doc_mat)
        tks_size = tks_chi.shape[0]
        
        trun_count = self._truncate_size(prop, top_k, tks_size)
        remain_tks = self._truncate(tks_chi, trun_count, vocab)
        
        if len(remain_tks) == 0:
            print("Warning, all examples tokens filtered")
            
        print(f"{1 - trun_count / tks_size:.3f}% tokens filtered with entropy alg")

        return remain_tks
    
    def _tks_chi(self, doc_label: np.array, doc_mat: np.array):
        """pass"""
        assert len(doc_mat.shape) == 2, "`doc_mat` must have shape of 2"
        
        # doc_size = doc_label.squeeze().shape[0]
        uni_doc_label = np.unique(doc_label, return_index=True)
        uni_doc_label = dict(zip(uni_doc_label[0], uni_doc_label[1]))  # for mapping unique label and index
        
        def _self_func(row_line):
            p_count, p_len = self._tk_count(row_line, doc_label, uni_doc_label)
            n_count, n_len = self._tk_count(1 - row_line, doc_label, uni_doc_label)
            chi_score = self._chi_score(p_count, n_count)
            return chi_score

        tks_chi = np.apply_along_axis(_self_func, 1, doc_mat)

        return np.asarray(tks_chi)
    
    
    def _count(self, labels: Union[np.array, List], doc_label_map: Dict) -> np.array:
        """pass"""
        labels = np.asarray(labels)
        label_count = np.zeros(len(doc_label_map))
        
        if labels.size == 0:
            return label_count
        
        label, count = np.unique(labels, return_counts = True)
        
        for lb, co in zip(label, count):
            label_count[doc_label_map[lb]] = co
        
        return label_count
    
    def _tk_count(self,
                  tk_label: Union[np.array, List],
                  doc_label: Union[np.array, List],
                  uni_doc_label: Dict):
        """pass"""
        tk_label = np.asarray(tk_label).squeeze()
        doc_label = np.asarray(doc_label)
        
        # filter zero count
        nzero_idx = np.where(tk_label != 0)
        
        pair_tk_doc = tk_label[nzero_idx] * doc_label[nzero_idx]
        return self._count(pair_tk_doc, uni_doc_label), len(nzero_idx[0]) # A, B, A+B
    
    def _chi_score(self,
                   p_count_mat: np.array,
                   n_count_mat: np.array):
        """pass"""
        total_count = np.sum([p_count_mat, n_count_mat])
        
        row_count = np.sum([p_count_mat, n_count_mat], axis = 1, keepdims = True)
        row_count /= total_count # [[3],[5]]
        
        col_val = np.vstack((p_count_mat, n_count_mat)) # [[1,2,3,4],[2,3,4,5]]
        col_count = np.repeat(col_val.sum(axis = 0, keepdims = True), repeats = 2, axis = 0)
        
        exp_val = col_count * row_count
        
        each_chi_score = (col_val - exp_val) ** 2 / exp_val # inf
        each_chi_score[each_chi_score == np.inf] = 0
        
        return np.sum(each_chi_score)
