"""
pass

1. 构建每句的情感词分
    1.1 程度词
    1.2 否定词
    1.3 停用词
    1.4 正 + 负 + 新词
    
2. 各类句式关系得分计算
"""
from typing import List, Text, Any, Dict, Union, Tuple
import numpy as np
from itertools import chain

from src import PUNC_PATTERN
from src.vocab import Vocab
from src.utils import Example, arrange_tokens_to_subseq, min_max_norm
from src.features import ngrams


class SentenceScore:
    
    name = ""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, *args, **kwargs):
        raise NotImplementedError("NotImplementedError")
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class TokenScore:
    
    name = ""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, *args, **kwargs):
        raise NotImplementedError("error")


class DenyExtremeTokenScore(TokenScore):
    """pass"""
    
    name = "deny_extreme_token_score"
    
    def __init__(self,
                 emo_dict: Vocab,
                 ext_dict: Vocab,
                 deny_dict: Vocab,
                 window: int = 3,
                 weight_deny: float = 0.5,
                 weight_extreme: float = 2,
                 ):
        super().__init__(SentenceScore)
        self.emo_dict = emo_dict
        self.ext_dict = ext_dict
        self.deny_dict = deny_dict
        self.window = max(1, window)
    
        self.weight_deny = weight_deny
        self.weight_extreme = weight_extreme
    
    def run(self,
            doc_tokens: List[List[Text]],
            tail_puncs: List = None,
            **kwargs):
        """pass"""
        return [self.run_sub_seq(tokens) for tokens in doc_tokens]
    
    def run_sub_seq(self,
                    tokens: List[Text]):
        """pass"""
        grams = ngrams(tokens, self.window)
        seq_score = np.zeros(shape = (len(grams)))
        
        for id, gram in enumerate(grams):
            if gram[-1] not in self.emo_dict.tk2idx:
                continue
            
            is_deny_first = False
            is_extreme_first = False
            
            _score = self.emo_dict.get_score(gram[-1])
            
            for tk in gram[:-1]:
                if tk in self.ext_dict.tk2idx:
                    if not is_deny_first:
                        is_extreme_first = True
                    _score *= self.ext_dict.get_score(tk)
                
                elif tk in self.deny_dict.tk2idx:
                    if not is_extreme_first:
                        is_deny_first = True
                    _score *= self.deny_dict.get_score(tk)
            
            weight = self.weight_deny if is_deny_first else self.weight_extreme
            _score *= weight
            
            seq_score[id] = _score
        
        return seq_score


class transitionSentenceScore(SentenceScore):
    
    name = "transition_sentence_score"
    
    def __init__(self,
                 weight_fw = 0.5,
                 weight_bw = 1.5):
        super().__init__()
        self.fw_tran = ["虽然", "虽说", "尽管"]
        self.bw_tran = ["但是", "不过", "但", "可是", "然而", ]
        
        self.weight_fw = weight_fw
        self.weight_bw = weight_bw

    def _load_fw_tran(self, tks: Union[Text, List[Text]]):
        if isinstance(tks, Text):
            tks = [Text]
        
        self.fw_tran += tks
    
    def _load_bw_tran(self, tks: Union[Text, List[Text]]):
        if isinstance(tks, Text):
            tks = [Text]
        
        self.bw_tran += tks
    
    def _unique_tran(self):
        self.fw_tran = list(set(self.fw_tran))
        self.bw_tran = list(set(self.bw_tran))
    
    def run(self,
            doc_tokens: List[List[Text]],
            tail_puncs: List[Text] = None,
            **kwargs):
        """pass"""
        self._unique_tran()
        sub_seq_score = np.ones(len(doc_tokens))
        
        for id, subseq in enumerate(doc_tokens):
            is_fw = False
            is_bw = False
            
            for tk in subseq:
                if tk in self.fw_tran:
                    is_fw = True
                elif tk in self.bw_tran:
                    is_bw = True
            
            if is_fw - is_bw == 1:
                sub_seq_score[id:] = self.weight_fw
            elif is_fw - is_bw == -1:
                sub_seq_score[id:] = self.weight_bw
        
        return sub_seq_score


class hypothesisSentenceScore(transitionSentenceScore):
    
    name = "hypothesis_sentence_score"
    
    def __init__(self):
        super().__init__()
        
        self.bw_tran = ["那么", "则", "也就是说", "也就是", "即"]


class tailpuncSentenceScore(SentenceScore):
    
    name = "tailpunc_sentence_score"
    
    def __init__(self):
        super().__init__()
        
        self.punc_mapping = {"?": -2, "!": 2, "？": -2, "！": 2, "。": 1}
        
    def _add_punc(self, char, score):
        self.punc_mapping[char] = score
        
    def run(self,
            doc_tokens: List[List[Text]],
            tail_puncs: List[Text], **kwargs):
        """pass"""
        sub_seq_score = np.ones(len(tail_puncs))
        
        _last_id = -1
        for id, punc in enumerate(tail_puncs):
            if punc in self.punc_mapping:
                sub_seq_score[(_last_id + 1):(id+1)] = self.punc_mapping[punc]
                _last_id = id
                
        return sub_seq_score


class totalSentenceScore(SentenceScore):
    
    name = "total_sentence_score"

    def __init__(self,
                 tok_method: Union[TokenScore, None] = None,
                 seq_methods: List[SentenceScore] = None,
                 weight = "mean"):
        super().__init__()
        
        self.tok_method = tok_method
        self.seq_methods = set(seq_methods or [])
        self.weight = self._is_weight_allows(weight)
    
    def add_method(self, methods: Union[SentenceScore,
                                        TokenScore,
                                        List[SentenceScore]]):
        """pass"""
        if isinstance(methods, SentenceScore):
            methods = [methods]

        [self.seq_methods.add(med) for med in methods]
    
    @property
    def weight_allows(self):
        return {"mean": np.mean,
                "max": np.max,
                "min": np.min,
                "mid": np.median}
    
    def run(self, example: Example, if_sign = False, if_norm = True, threshold = 0):
        """pass"""
        exam_tokens = example.get("tokens")
        if not exam_tokens:
            return 0.
        
        doc_tokens, tail_puncs = arrange_tokens_to_subseq(example.get("tokens"))
        
        assert self.tok_method and isinstance(self.tok_method, TokenScore), "a `TokenScore` method must exist"
        seq_methods = list(self.seq_methods)
        if not self.seq_methods:
            seq_methods = [lambda x, y: np.ones(shape = len(x))]
        
        tok_score = self.tok_method.run(doc_tokens, tail_puncs)
        
        seq_scores = []
        for med in seq_methods:
            seq_score = med(doc_tokens, tail_puncs)
            seq_score = self._method_score(token_score_seq = tok_score,
                                           seq_score_seq = seq_score,
                                           if_norm = if_norm)
            seq_scores.append(seq_score)
            
        seq_scores = np.asarray(seq_scores)
 
        seq_scores = self.weight.__call__(seq_scores)
        if if_sign:
            seq_scores = np.sign(seq_scores)
        # seq_scores = np.where(seq_scores > threshold, 1, -1)
        
        return seq_scores
        
    def _method_score(self,
                      token_score_seq: List[float],
                      seq_score_seq: Union[np.array, List, None] = None,
                      if_norm = True):
        """pass"""
        seq_score_seq = np.ones(shape = len(token_score_seq)) if seq_score_seq is None else seq_score_seq
        seq_score = np.asarray(list(chain.from_iterable(map(lambda x, y: x * y, token_score_seq, seq_score_seq))))
        
        if if_norm:
            seq_score = min_max_norm(seq_score)

        return seq_score
    
    def _is_weight_allows(self, weight: Text):
        if weight in self.weight_allows:
            return self.weight_allows.get(weight)
        
        else:
            raise ValueError(f"{weight} not allowed, only `mean`, `max`, `min`, `mid` allowed")
