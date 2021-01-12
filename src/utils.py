from typing import List, Text, Any, Dict, Union, Tuple
from collections import defaultdict
import codecs
import csv
import re
import numpy as np

from src import ZH_PATTERN, PUNC_PATTERN

class Example:
    def __init__(self, text: Text, label: Union[float, int, Text]):
        self.text = text
        self.label = label
        self.data = {}
        
    def set(self, prop, value):
        self.data[prop] = value
        
    def get(self, prop):
        return self.data.get(prop, None)

def read_line_from_txt(fname):
    with codecs.open(fname, encoding = "utf-8") as f:
        lines = f.read().strip().split()
        
        return lines

def read_data_from_csv(fname = "./corpus/weibo_senti_100k.csv",
                       label_map = {"1": 1, "0": -1}):
    examples = []

    with codecs.open(fname, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')

        col_names = None
        for id, line in enumerate(reader):
            if id == 0:
                col_names = line[2:]
                continue
            
            exam = Example(text = line[1],
                           label = label_map.get(line[0], line[0]) if label_map else line[0])
            if col_names:
                for col, val in zip(col_names, line[2:]):
                    exam.set(col, val)
                    
            examples.append(exam)

        return examples

def write_data_to_csv(fname: Text,
                      examples: List[Example]) -> None:
    """pass"""
    with codecs.open(fname, "w", encoding = "utf-8") as f:
        writer = csv.writer(f, delimiter=',')
        
        writer.writerow(["label", "text", *examples[0].data])
        for exam in examples:
            writer.writerow([exam.label, exam.text, *exam.data.values()])
        

def label_distribution_viewer(examples: List[Example],
                              label_map = None,
                              verbose = True):
    label_map = label_map or {}
    
    label_count = defaultdict(int)
    
    for exam in examples:
        label_count[label_map.get(exam.label, exam.label)] += 1
        
    if verbose:
        for label, count in label_count.items():
            print(f"{label}:\t{count}")
    
    return label_count


def emoji_distribution_viewer(examples: List[Example],
                              min_df = 1,
                              max_df = 5,
                              drop_df = 2,
                              verbose = True):
    assert min_df <= max_df, "`min_df` must less than `max_df`"
    
    pat = re.compile("\[%s{%d,%d}\]" % (ZH_PATTERN, min_df, max_df))
    emoji_count = defaultdict(int)
    
    for exam in examples:
        _res = pat.findall(exam.text)
        for x in _res:
            emoji_count[x] += 1
        
    emoji_count = {k: v for k, v in emoji_count.items() if v > drop_df}
    emoji_count = dict(sorted(emoji_count.items(), key = lambda x: x[1], reverse = True))
    
    print(f"finding emoji label {len(emoji_count)}")
    
    if verbose:
        for id, (label, count) in enumerate(emoji_count.items()):
            if id >= 10:
                break
                
            print(f"{label}:\t{count}")

    return emoji_count

def arrange_tokens_to_subseq(tokens: List[Text]) -> Tuple[List[List[Text]], List[Text]]:
    """pass"""
    if not tokens:
        return [], []
    
    punc_pat = re.compile(f"^{PUNC_PATTERN}")
    seq_tokens = []
    sub_tokens = []
    tail_puncs = []

    for token in tokens:
        if punc_pat.search(token):
            if sub_tokens:
                seq_tokens.append(sub_tokens)
                tail_puncs.append(token)
        
            sub_tokens = []
    
        else:
            sub_tokens.append(token)

    if sub_tokens:
        seq_tokens.append(sub_tokens)
        tail_puncs.append("。")

    assert len(seq_tokens) == len(tail_puncs), "bug exists, plz return output"

    return seq_tokens, tail_puncs

def min_max_norm(score: Union[np.array, List]):
    """pass"""
    score = np.asarray(score).flatten()
    
    # lenght of 1
    if score.size == 1:
        return np.sign(score)
    # all element equal
    if (score == score[0]).all():
        return np.sign(score[0])
    
    return (score - score.min()) / (score.max() - score.min())
    
# if __name__ == '__main__':
#     examples = read_data_from_csv("./corpus/new.txt")
#     label_distribution_viewer(examples)
    