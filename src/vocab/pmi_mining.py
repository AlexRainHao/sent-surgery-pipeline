"""pass"""

from typing import List, Text, Any, Dict, Union
from collections import Counter, defaultdict
from itertools import chain, product
import re
from tqdm import tqdm
import numpy as np

from src import PUNC_PATTERN
from src.utils import Example, write_data_to_csv
from src.vocab import Vocab
from src.features import doc_onehot_mat, doc_label_mat, token_emotion_mat, ngrams

__all__ = ["pmi_score", "pair_pmi", "create_vocab", "seedWordMining", "spanNewWordMining"]

def pmi_score(co_count: Union[np.array, int],
              base_count: Union[np.array, int],
              alter_count: Union[np.array, int]):
    """psss"""
    if isinstance(co_count, int):
        doc_size = 1
    elif len(co_count.shape) == 0:
        doc_size = co_count.size
    else:
        doc_size = co_count.shape[-1]
    
    return np.log((co_count + 1) / doc_size / (base_count * alter_count + 1))

def pair_pmi(doc_mat: Union[np.array, List[List]],
             emo_mat: Union[np.array, List],
             vocab: Vocab,
             if_sign=True) -> List[Example]:
    """pass"""
    seed_idx, pos_idx, neg_idx = vocab.get_all_group()
    
    doc_mat = np.asarray(doc_mat).squeeze()
    emo_mat = np.asarray(emo_mat).flatten()
    
    if len(doc_mat.shape) != 2:
        raise ValueError("doc_mat must have dimension of 2")
    
    pair_seed_pos_idx = list(product(seed_idx, pos_idx))
    pair_seed_neg_idx = list(product(seed_idx, neg_idx))
    
    scores = defaultdict(float)
    
    pbar = tqdm(total = len(pair_seed_neg_idx) + len(pair_seed_pos_idx), desc = "so pmi annotation calling")
    for group in [pair_seed_pos_idx, pair_seed_neg_idx]:
        for seed, emo in group:
            sub_count = doc_mat[[seed, emo]].sum(axis=1)
            co_curr = (sub_count == 2).sum()
            seed_curr, emo_curr = sub_count
            scores[seed] += pmi_score(co_curr, seed_curr, emo_curr) * emo_mat[emo]
            pbar.update(1)
    pbar.close()
    
    scores = np.asarray([scores[idx] for idx in seed_idx])
    scores = np.sign(scores) if if_sign else scores
    
    return [Example(text=vocab.get_tk(tk), label=sco) for tk, sco in zip(seed_idx, scores)]

def create_vocab(pos_words: Union[List[Text], Vocab],
                 neg_words: Union[List[Text], Vocab],
                 seed_words: Union[List[Text], Vocab]) -> Vocab:
    """pass"""
    
    _convert_func = lambda x: list(x.tk2idx.keys()) if isinstance(x, Vocab) else x
    
    pos_words = _convert_func(pos_words)
    neg_words = _convert_func(neg_words)
    seed_words = _convert_func(seed_words)
    
    emo_vocab = Vocab()
    emo_vocab.add_seq(pos_words, emo_vocab.postive_name, 1)
    emo_vocab.add_seq(neg_words, emo_vocab.negtive_name, -1)
    emo_vocab.add_seq(seed_words, emo_vocab.alters_name, 0)
    return emo_vocab
    
    
class seedWordMining:
    
    name = "seed_word_mining"
    
    def __init__(self,
                 stop_words: Union[List[Text], Vocab] = None,
                 base_pos_words: Union[List[Text], Vocab] = None,
                 base_neg_words: Union[List[Text], Vocab] = None):
        """
        Parameters
        ----------
        examples : List[Example], token list for each sequence of doc
        stop_words: Union[List[Text], None], stop words list
        base_neg_words : Union[List[Text], None], base negative words
        base_pos_words : Union[List[Text], None], base positive words
        """
        self.stop_words = Vocab.gene_from_list(stop_words, score = 0) if \
            isinstance(stop_words, List) else stop_words
        self.base_pos_words = Vocab.gene_from_list(base_pos_words, name = Vocab().postive_name, score = 1) if \
            isinstance(base_pos_words, List) else base_pos_words
        self.base_neg_words = Vocab.gene_from_list(base_neg_words, name = Vocab().negtive_name, score = -1) if \
            isinstance(base_neg_words, List) else base_neg_words
        
        self.seedwords = None
    
    def run(self,
            examples: List[Example],
            min_df: int = 3,
            min_prop: float = 0.5,
            min_len: int = 2,
            top_k: int = None,
            if_tag: bool = False) -> List[Example]:
        """
        select seed words from corpus based on token frequency,
        that will used to calculate PMI score with new mining word

        the seed words ARE NOT ANNOTATION

        Parameters
        ----------
        
        examples: List[Example], examples with `token` property
        min_df: int, minimum frequency
        min_prop: float, proportion of tokens for consideration
        min_len: minimum token length
        top_k: int, choose most common token
        """
        doc_tokens = [exam.get("tokens") for exam in examples]
        
        punc_pat = re.compile(f"{PUNC_PATTERN}+")
        
        counter = Counter(chain.from_iterable(doc_tokens))
        
        min_prop = 0.1 if min_prop > 1 or min_prop < 0 else min_prop
        min_count = int(len(counter) * min_prop)
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:min_count]
        
        seedwords = list(filter(lambda x: x[1] > min_df and
                                          not punc_pat.search(x[0]) and
                                          len(x[0]) > min_len and
                                          x[0] not in self.stop_words.tk2idx,
                                counter))
        
        top_k = top_k or len(seedwords)
        seedwords = seedwords[:top_k]
        
        print(f"mining total seed word {len(seedwords)}")
        
        if if_tag:
            self.seedwords = self.anno_seed_word(doc_tokens, seedwords)
        
        else:
            self.seedwords = [Example(text=tk, label=0) for tk in seedwords]
        
        return self.seedwords
    
    def anno_seed_word(self,
                       doc_tokens: List[List[Text]],
                       seed_words: List[Text]) -> List[Example]:
        """auto annotation for seed words selected through `PMI`,
        where `so_pmi` would calculated from
            so_pmi(word) = mean(PMI(word, Pw)) - mean(PMI(word, Nw))

        if so_pmi(word) > 0, the seed word would tagged as positive
        if so_pmi(word) = 0, tagged as neutral
        if so_pmi(word) < 0, tagged as negative
        """
        _seed_words_vocab = Vocab.gene_from_list(seed_words, Vocab().alters_name, 0)
        emo_vocab = self.base_pos_words + self.base_neg_words + _seed_words_vocab
        
        emo_mat = token_emotion_mat(emo_vocab)
        doc_mat = doc_onehot_mat(doc_tokens, emo_vocab)
        
        so_pmi_score = pair_pmi(doc_mat, emo_mat, emo_vocab)
        
        return so_pmi_score
    
    def _to_csv(self, fname: Text):
        """pass"""
        if not self.seedwords:
            print("no mining seedwords found")
        
        else:
            write_data_to_csv(fname, self.seedwords)

class spanNewWordMining:
    
    name = "span_new_word_mining"
    
    def __init__(self,
                 examples: List[Example],
                 seed_tokens: List[Example],
                 extreme_words: Union[List[Text], Vocab],
                 deny_words: Union[List[Text], Vocab],
                 base_pos_words: Union[List[Text], Vocab] = None,
                 base_neg_words: Union[List[Text], Vocab] = None,
                 ):
        """
        Parameters
        ----------
        examples: List[Example], each example could extract `tokens`
        seed_tokens: List[Example], seed tokens mined
        extreme_words: List[Text], a set of extreme words
        deny_words: List[Text], a set of deny words
        base_pos_words: base positive words if needed
        base_neg_words: base negative words if needed
        """
        
        self.doc_tokens = [exam.get("tokens") for exam in examples]
        self.doc_labels = [exam.label for exam in examples]
        self.doc_size = len(self.doc_tokens)
        
        self.seed_tokens = seed_tokens
        self.alia_pos_words, self.alia_neg_words = self._alia_emo_words()
        
        self.extreme_words = Vocab.gene_from_list(extreme_words, score = 2) if isinstance(extreme_words, List) else extreme_words
        self.deny_words = Vocab.gene_from_list(deny_words) if isinstance(deny_words, List) else deny_words
        self.span_words = self.extreme_words + self.deny_words # vocab
        
        self.base_pos_words = Vocab.gene_from_list(base_pos_words, name = Vocab().postive_name, score = 1) if \
            isinstance(base_pos_words, List) else base_pos_words
        self.base_neg_words = Vocab.gene_from_list(base_neg_words, name = Vocab().negtive_name, score = -1) if \
            isinstance(base_neg_words, List) else base_neg_words
        
        self.alter_tks = None
        self.new_tks = None
    
    def _alia_emo_words(self):
        """extract seed words emotion label"""
        pos, neg = [], []
        
        for tk_exam in self.seed_tokens:
            if tk_exam.label > 0:
                pos.append(tk_exam.text)
            else:
                neg.append(tk_exam.text)
        
        return pos, neg
    
    def _single_span_token(self,
                           doc_token: List[Text],
                           window: int = 2):
        """extract window tokens after extreme words or deny words for each doc"""
        
        if window <= 1:
            return Counter()
        
        grams = ngrams(doc_token, window)
        
        win_tks = map(
            lambda gram: list(filter(
                lambda tk: tk not in self.span_words.tk2idx, gram[1:]
            )) if gram[0] in self.span_words.tk2idx else [], grams
        )
        
        return Counter(chain.from_iterable(win_tks))
    
    def run(self,
            min_window: int = 2,
            max_window: int = 4,
            min_len: int = 2,
            min_count: int = 1,
            prop: float = 1.,
            top_k: Union[None, int] = None,
            if_tag: bool = True,
            alia_base_emo: bool = True):
        """
        extract window tokens as suspicious new span tokens,
        a series parameters could determine truncation
        
        Parameters
        ----------
        min_len: minimum length of token
        min_count: minimum frequency of token
        prop: truncate a proportion of set
        top_k: only top K token output
        """
        assert min_window <= max_window, "window range error"
        
        alter_tks = Counter()
        
        pbar = tqdm(total=self.doc_size * (max_window - min_window + 1),
                    desc="span word search")
        for win in range(min_window, max_window + 1):
            for doc_token in self.doc_tokens:
                alter_tks += self._single_span_token(doc_token=doc_token,
                                                     window=win)
                pbar.update(1)
        pbar.close()
        
        min_len = max(0, min_len)
        min_count = max(0, min_count)
        prop = 1. if prop < 0 else min(1., max(0., prop))
        top_k = top_k if top_k and top_k > 0 else len(alter_tks)
        trun_count = min(int(len(alter_tks) * prop), top_k)
        
        alter_tks = sorted(alter_tks.items(), key=lambda x: x[1], reverse=True)
        alter_tks = list(filter(lambda x: x[1] > min_count and len(x[0]) > min_len, alter_tks))[:trun_count]
        print(f"mining suspect span tokens {len(alter_tks)}")
        
        self.alter_tks = alter_tks
        
        if if_tag:
            return self.anno_mining_token(alia_base_emo)
        
        else:
            return alter_tks
    
    def anno_mining_token(self, alia_base_emo=True) -> List[Example]:
        """
        use `SO_PMI` and `Doc_Distance` to annotate suspicious mining span tokens
        
        where `Doc_Distance` as:
            doc_dist(w) = NDoc_pos(w) - NDoc_neg(w)
        
        Parameters
        ----------
        alia_base_emo: bool, determine whether use base emotion vocab
        """
        alter_tks = [x[0] for x in self.alter_tks]
        
        pos_words = Vocab.gene_from_list(self.alia_pos_words, Vocab().postive_name, 1)
        neg_words = Vocab.gene_from_list(self.alia_neg_words, Vocab().negtive_name, -1)
        
        if alia_base_emo:
            pos_words += self.base_pos_words
            neg_words += self.base_neg_words
        
        emo_vocab = pos_words + neg_words
        
        # filter base emo tokens
        alter_tks = [tk for tk in alter_tks if tk not in emo_vocab.tk2idx]
        
        # create each mat
        emo_vocab += Vocab.gene_from_list(alter_tks,
                                          name = Vocab().alters_name,
                                          score = 0)
        
        emo_mat = token_emotion_mat(emo_vocab)
        label_mat = doc_label_mat(self.doc_labels)
        doc_mat = doc_onehot_mat(self.doc_tokens,
                                 emo_vocab)
        
        alter_idx = emo_vocab.get_group(emo_vocab.alters_name)
        
        # so_pmi
        so_pmi_scores_obj = pair_pmi(doc_mat, emo_mat, emo_vocab)
        so_pmi_scores = [exam.label for exam in so_pmi_scores_obj]
        
        # doc_distance
        doc_dist = np.sum(doc_mat[alter_idx] * label_mat, axis=1)
        pmi_dist_scores = so_pmi_scores * doc_dist
        
        # only score greater than 0 selected
        res_idx = np.where(pmi_dist_scores > 0)[0]
        res_exam = [so_pmi_scores_obj[idx] for idx in res_idx]
        
        print(f"mining new span token {len(res_exam)}")
        
        self.new_tks = res_exam
        
        return res_exam
    
    def _to_csv(self, fname: Text):
        """pass"""
        if not self.new_tks:
            print("no mining span tokens found")
        
        else:
            write_data_to_csv(fname, self.new_tks)

