"""
* 去除 人名
* 去除 英文
* 去除 url
* 去除 标签
* 连续标点归一（除了[])
* emoji归一
"""

from typing import Union, List
import re
from src import ZH_PATTERN, PUNC_PATTERN
from src.utils import Example


class Cleaner:
    """pass"""
    
    name = ""
    
    pattern = None
    
    component_config = None
    
    def run(self, example: Example, **kwargs) -> None:
        self.read_args()
        
        if not self.pattern:
            return
        
        text = example.text
        for pat in self.pattern:
            text = re.compile(pat[0]).sub(pat[1], text)
            
        example.text = text
    
    def read_args(self):
        if not self.component_config:
            return
        
        for key, val in self.component_config.items():
            self.__setattr__(key, val)
    

class removeShortSeq(Cleaner):
    """pass"""

    name = "remove_short_sequence"
    
    component_config = {"min_len": 5}
    
    def run(self, example: Example, **kwargs):
        self.read_args()
        
        text = example.text
        if len(text) < self.min_len:
            example.text = ""


class removeSpace(Cleaner):
    """pass"""
    
    name = "remove_space"
    
    pattern = [("\s+", "")]

    
class removeName(Cleaner):
    """pass"""
    
    name = "remove_name_pattern"
    
    pattern = [("(//)?(回复)?@[^:|：|\ .]*", "")]
    # pattern = [(f"(//)?(回复)?@.*?{PUNC_PATTERN}", "")]
    
class removeHttpUrl(Cleaner):
    """pass"""
    
    name = "remove_http_url_pattern"
    
    pattern = [("https?[a-zA-Z0-9/%.:]+", "")]
    
    
class removeTags(Cleaner):
    """pass"""
    
    name = "remove_tags"
    
    pattern = [("#.*?#", "")]
    
    
class removeEng(Cleaner):
    """pass"""
    
    name = "remove_english"
    
    pattern = [("[a-zA-Z0-9]{2,}", "")]
    

class normConsecPunc(Cleaner):
    """pass"""
    
    name = "norm_consecutive_punc"
    
    
    pattern = [(f"({PUNC_PATTERN}+)({PUNC_PATTERN})", "\\g<2>"),
               (f"^{PUNC_PATTERN}", "")]
    

class normConsecEmoji(Cleaner):
    """pass"""
    
    name = "norm_consecutive_emoji"
    
    pattern = f"\[{ZH_PATTERN}+\]"
    
    def run(self, example: Example, **kwargs):
        text = example.text
        
        text_list = re.compile(self.pattern).split(text)
        emoji_list = re.compile(self.pattern).findall(text)
        
        example.text = ''.join(text_list)
        
        example.set("emoji", list(set(emoji_list)))
        

class wrapperFullClearer():
    """pass"""
    
    name = "simple_wrapper_full_cleaner"
    
    ops = [removeShortSeq(), removeHttpUrl(), removeName(), removeSpace(),
           removeTags(), removeEng(), normConsecEmoji(), normConsecPunc()]
    
    def run(self, example: Union[Example, List[Example]], **kwargs):
        if isinstance(example, Example):
            example = [example]

        for exam in example:
            for op in self.ops:
                op.run(exam)

if __name__ == '__main__':
    text = Example("@张晓鹏jonathan 土耳其的事要认真对待[哈哈]，否则直接开除。@丁丁看世界 很是细心，酒店都全部OK啦。", 1)
    wrapperFullClearer().run(text)
    print(text.text)
    