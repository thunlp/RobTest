import random
from copy import deepcopy
from turtle import distance
from .anthro.anthro_lib import ANTHRO
from ..rule_transform import RuleTransform
import numpy as np
from evaluator import EditDistance
import time
import math
import torch
import os
anthro = ANTHRO()
anthro.load('src/AttackMethod/RuleBased/Char/anthro/ANTHRO_Data_V1.0')
class PhoneticTransform(RuleTransform):
    def __init__(self, degree, aug_num=1,distance_type="char",ori_sent=""):
        super().__init__(degree, aug_num)
        self.dis_type = distance_type
        self.distance = EditDistance()
        self.anthro = anthro
        self.degree = degree
        self.ori_sent = ori_sent




    def transform(self, sentence, case_sensitive=True):
        
        sents = []
        split = sentence.strip().split()
        sent_len = len(sentence.strip())
        if self.dis_type=="char":
            sent_len = len(sentence.strip())
        else:
            sent_len = len(split)
        # start = time.time()
        for _ in range(self.aug_num):
            # if self.dis_type == "char":
            #     indices = np.random.choice(len(sentence), len(sentence), replace=False)
            # else:
            indices = np.random.choice(len(sentence.split()), len(sentence.split()), replace=False)
         
            # indices = np.random.choice(len(split), len(split), replace=False)
            word_split = deepcopy(split)
            
            for idx in indices:
                if self.distance(" ".join(split), " ".join(word_split))/sent_len >= self.degree:
                    break     
                word = word_split[idx]
                
                substitude_word = self.phonetic_transform(word, case_sensitive=case_sensitive)
                
                # end = time.time()
                # print(end-start)
                if substitude_word is None:
                    continue
                word_split[idx] = substitude_word

 
            sents.append(" ".join(word_split))
        # end = time.time()
        # print(end-start)
        # torch.save(self.candidate_words_dict,phonetic_dict_path)
        return sents


    def phonetic_transform(self, word, case_sensitive=True):
        distance = random.randint(1,len(word)) if self.dis_type=='char' else 1
        candidate_words = list(self.anthro.get_similars(word, level=1, distance=distance, strict=True))

        # if not case_sensitive:
        #     candidate_words = list(set([w.lower() for w in candidate_words]))
        if word in candidate_words:
            candidate_words.remove(word)
        if len(candidate_words) > 0:
            return random.choice(candidate_words)
        else: 
            return None


    def transform_target(self, sent,sent_idx, case_sensitive=True):
        # print(sent_idx)
        sentence_tokens = sent.split()
        ori_sentence_tokens = self.ori_sent.split()
        # try:
        word = ori_sentence_tokens[sent_idx]
        # self.trans = self(word)
        substitute = self.phonetic_transform(word, case_sensitive=case_sensitive)
        if substitute == None:
            return None
        
        sentence_tokens[sent_idx] = substitute

        return ' '.join(sentence_tokens)


    def transform_char(self, sent,sent_idx, case_sensitive=True):
        # print(sent_idx)
        sentence_tokens = sent.split()
        # try:
        word = sentence_tokens[sent_idx]
        substitute = self.phonetic_transform(word, case_sensitive=case_sensitive)
        if substitute == None:
            return None
        
        sentence_tokens[sent_idx] = substitute

        return ' '.join(sentence_tokens)
