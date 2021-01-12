"""
a series of tokenizer
"""

from typing import List, Dict, Any, Text, Union
import importlib
from tqdm import tqdm
from glob import glob
import os.path as opt
from utils import Example

class Tokenizer:
    name = ""
    
    component_config = {"dictionary": None,
                        "tokens": None}
    
    def __init__(self,
                 component_config: Dict = None,
                 **kwargs):
        if component_config:
            self.component_config.update(component_config)
        
        self.import_pkg_from_list()
        
        self.load_dict()

    def load_dict(self, **kwargs):
        pass
    
    def run(self, examples: List[Example], **kwargs):
        for exam in tqdm(examples, desc = "tokenizing"):
            self.srun(exam)
        
    
    def srun(self, example: Example):
        raise NotImplementedError("NotImplementedError")
    
    @staticmethod
    def required_pkg():
        return []
    
    def import_pkg_from_list(self):
        """pass"""
        for pkg in self.required_pkg():
            try:
                op = importlib.import_module(pkg)
                globals()[pkg] = op
            except:
                raise ImportError(f"pkg {pkg} not found")
        
    def load_component_from_dict(self, *args, **kwargs):
        self.component_config.update(*args)
        self.component_config.update(**kwargs)
    
        
class jiebaTokenizer(Tokenizer):
    name = "jieba_tokenizer"
    
    component_config = {"dictionary": None,
                        "tokens": None}

    def __init__(self,
                 component_config: Dict = None):
        super().__init__(component_config)

    @staticmethod
    def required_pkg():
        return ["jieba"]
    
    def srun(self, example: Example, **kwargs) -> None:
        tokens = jieba.lcut(example.text)
        
        example.set("tokens", tokens)
    
    
    def load_dict(self, **kwargs):
        dict = self.component_config.get("dictionary", []) or []
        tokens = self.component_config.get("tokens", []) or []
        
        if not dict:
            dict = []
        
        elif opt.isdir(dict):
            dict = glob("{}/*".format(dict))
            
        elif opt.isfile(dict):
            dict = [dict]
        
        for file in dict:
            try:
                jieba.load_userdict(file)
            except:
                print(f"error in load dictionary {file}, loading would ignored")

        for token in tokens:
            try:
                jieba.add_word(token)
            except:
                print(f"error in load token {token}, loading would ignored")


class lacTokenizer(Tokenizer):
    name = "lac_tokenizer"
    
    component_config = {"dictionary": None,
                        "sep": "\n"}

    def __init__(self,
                 component_config: Dict = None,
                 **kwargs):
        # super().__init__(component_config)
        if component_config:
            self.component_config.update(component_config)
            
        self.import_pkg_from_list()
        self.lac = LAC.LAC(mode = "seg")
        self.load_dict()
    
    @staticmethod
    def required_pkg():
        return ["LAC"]
    
    def srun(self, example: Example, **kwargs) -> None:
        tokens = self.lac.run(example.text)
    
        example.set("tokens", tokens)
    
    def load_dict(self, **kwargs):
        dict = self.component_config.get("dictionary", []) or []
        
        if not dict:
            dict = []
        
        elif opt.isdir(dict):
            dict = glob("{}/*".format(dict))
            
        elif opt.isfile(dict):
            dict= [dict]
            
        for file in dict:
            try:
                self.lac.load_customization(file,
                                            sep = self.component_config.get('sep', "\n"))
            except:
                print(f"error in load dictionary {file}, loading would ignored")
                
                