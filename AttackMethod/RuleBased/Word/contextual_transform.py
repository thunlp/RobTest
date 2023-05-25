from ..rule_transform import RuleTransform
from transformers import pipeline
from copy import deepcopy
import random
import numpy as np
from nltk.corpus import wordnet as wn
import torch
import math
import os
# sent_dict_path = os.path.join(os.path.dirname(__file__), 'contextual_jigsaw_dict.pt')
# sent_dict = torch.load(sent_dict_path) if os.path.exists(sent_dict_path) else {}
mlm_ckpt='bert-base-uncased'
thefilter = pipeline('fill-mask', model=mlm_ckpt, tokenizer=mlm_ckpt)
class ContextualTransform(RuleTransform):
    

    def __init__(self, degree, aug_num=1,ori_sent=''):
        super().__init__(degree, aug_num)
        
        self.ori_sent=ori_sent
        self.sents = [ori_sent]
        

        # self.sent_dict = sent_dict
        if len(ori_sent)>0:

            self.filler = thefilter
            self.total_fill_results = self.get_fill_results(ori_sent)
            # torch.save(self.sent_dict, sent_dict_path)
        else:
            self.filler = thefilter
            




    def transform(self, sentence):    
        self.ori_sent = sentence   

        word_split = sentence.split()

        self.total_fill_results = self.get_fill_results(sentence)

        substitute_num = math.ceil(len(word_split)*self.degree)
        self.sents = [sentence]
        for _ in range(self.aug_num):
            count = 0
            indices = np.random.choice(len(word_split), len(word_split), replace=False)
            candidate_sent = sentence
            # iteratively substitute words in sentence, the iteration number is determined by self.degree
            for idx in indices:
                if count >= substitute_num:
                    break
                candidate_sent = self.mlm_substitute(candidate_sent.strip(), idx,self.total_fill_results)
                if candidate_sent is None:
                    candidate_sent = self.sents[-1]
                    continue
                count += 1
            if candidate_sent not in self.sents:
                self.sents.append(candidate_sent)
        if len(self.sents)>1:
            self.sents.remove(sentence)
        # torch.save(self.sent_dict, sent_dict_path)
        return self.sents    


    def mlm_substitute(self, sentence, idx,total_fill_results):

        split = sentence.split()
        # mask-and-fill
        fill_results = total_fill_results[idx]
        mask_word = self.ori_sent.split()[idx]

        for fill_result in fill_results:
            split_temp = deepcopy(split)
            pred_word = fill_result['token_str']
            split_temp[idx] = pred_word
            candidate_sent = ' '.join(split_temp)
            # deduplicate
            if candidate_sent.lower() == self.ori_sent.lower() \
                or candidate_sent in self.sents:
                continue
            is_antonym = self.check_antonyms(mask_word, pred_word)
            if is_antonym:
                continue

            return candidate_sent



    def get_fill_results(self,sentence):
        word_split = sentence.split()
        total_fill_results = []
        for i in range(len(word_split)):
            split = deepcopy(word_split)
            split[i] = self.filler.tokenizer.mask_token
            fill_results = self.filler(' '.join(split), top_k=10) # generate k results
            total_fill_results.append(fill_results)
        return total_fill_results
    # def check_pos_tag(self, ori_sent, adv_sent, idx): # return bool
    #     try:
    #         ori_pos_tags = self.pos_tagger.tag(ori_sent.split())
    #         adv_pos_tags = self.pos_tagger.tag(adv_sent.split())
    #         ori_tag = ori_pos_tags[idx][1]
    #         adv_tag = adv_pos_tags[idx][1]
    #         return True if ori_tag == adv_tag else False
    #     except:
    #         return False


    def get_synonyms(self, mask_word):

        synonyms = set()
        for syn in wn.synsets(mask_word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        if mask_word in synonyms:
            synonyms.remove(mask_word)

        return list(synonyms)

    def get_word_antonyms(self, word):
        antonyms_lists = set()
        try:
            for syn in wn.synsets(word):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms_lists.add(l.antonyms()[0].name())
            return list(antonyms_lists)
        except:
            return []

    def check_antonyms(self, mask_word, pred_word):
            # filter antonyms
        antonyms_list = self.get_word_antonyms(mask_word)
        return True if pred_word in antonyms_list else False

    def transform_target(self, sent,sent_idx):
        ori_sentence_tokens = self.ori_sent.split()
        # sentence_tokens = sent.split()
        # sentence_tokens[sent_idx] = ori_sentence_tokens[sent_idx]
        return self.mlm_substitute(sent.strip(), sent_idx,self.total_fill_results)