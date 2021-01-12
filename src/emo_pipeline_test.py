import random

# ================
# test for load data
from src.utils import read_data_from_csv, read_line_from_txt, Example
from src.vocab import Vocab
dataset = read_data_from_csv("../corpus/weibo_senti_100k.csv", label_map = {"1": 1, "0": -1})
dataset = dataset[:10] + dataset[-10:]
dataset_size = len(dataset)

stop_word_dict = read_line_from_txt("../dict/stopword.txt")
base_posword_dict = read_line_from_txt("../dict/ntusd/NTUSD_positive.txt")
base_negword_dict = read_line_from_txt("../dict/ntusd/NTUSD_negative.txt")
extreme_words = read_line_from_txt("../dict/hownet/extreme.txt")
extreme_word_dict = Vocab()
for word in extreme_words:
    if not word.startswith("-"):
        _lb, _tt = word.strip().split(',')
        _lb = int(_lb)
        extreme_word_dict.add(_tt, _lb)
deny_words_dict = read_line_from_txt("../dict/deny.txt")

# =================
# # test for label\emoji distribution
# from src.utils import label_distribution_viewer, emoji_distribution_viewer
# label_distribution_viewer(dataset, label_map={"1": "pos", "0": "neg"}, verbose=True)
# emoji_distribution_viewer(dataset, drop_df = 5)

# =================
# test normalizer
from src.normalizer import wrapperFullClearer
clean_op = wrapperFullClearer()
clean_op.run(dataset)

# TODO: emoji score annotation

# =================
# test tokenizer
from src.tokenizer import jiebaTokenizer
jieba_op = jiebaTokenizer()
jieba_op.run(dataset)

# lac_op = lacTokenizer()
# lac_op.run(dataset)

# ==================
# test vocab mining
# 1.1 seed word mining
from src.vocab.pmi_mining import seedWordMining, spanNewWordMining

# seed_word_op = seedWordMining(stop_word_dict, base_posword_dict, base_negword_dict)
# seed_words = seed_word_op.run(dataset, if_tag = True, top_k = 30) # Example

# seed_words = [Example(text = "哈哈哈", label = 1), Example(text = "卧槽", label = -1)]
# # 1.2 field new word mining
# new_word_op = spanNewWordMining(dataset,
#                                 seed_words,
#                                 extreme_word_dict,
#                                 deny_words_dict,
#                                 base_posword_dict,
#                                 base_negword_dict)
# new_word_op.run(min_window = 2, max_window = 3, alia_base_emo = False)

# ==================
# test for sentence score
from src.sentence_score.sentence_score import *

pos_emo_dict = Vocab.gene_from_list(base_posword_dict, name = Vocab.postive_name, score = 1)
neg_emo_dict = Vocab.gene_from_list(base_negword_dict, name = Vocab.negtive_name, score = -1)
deny_emo_dict = Vocab.gene_from_list(deny_words_dict, score = 0)
ext_emo_dict = extreme_word_dict
emo_dict = pos_emo_dict + neg_emo_dict

score_op = totalSentenceScore(tok_method = DenyExtremeTokenScore(emo_dict = emo_dict,
                                                                 ext_dict = ext_emo_dict,
                                                                 deny_dict = deny_emo_dict),
                              seq_methods = [transitionSentenceScore(),
                                             hypothesisSentenceScore(),
                                             tailpuncSentenceScore()])

for data in dataset:
    print(score_op.run(data))


