from click import prompt
from ..rule_transform import RuleTransform
# import distance
import random
import numpy as np
from copy import deepcopy
import math

# # By keyboard proximity
# copy from textbugger
keyboard_neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk", ";":"op[l',./", '\'':"p[];./",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn",
            ',': "mkl.", ".": ",l;/", 
        }
keyboard_neighbors_keys = keyboard_neighbors.keys()


class TypoTransform(RuleTransform):
    def __init__(self, degree, aug_num=1,distance_type="char",ori_sent=""):
        super().__init__(degree, aug_num)
        self.transform_types = ['delete', 'insert', 'replace', 'swap', 'repeat']
        self.TRANSFORMATION = {
            'delete': self.delete_transform, 
            'insert': self.insert_transform, 
            'replace': self.replace_transform,
            'swap': self.swap_transform, 
            'repeat': self.repeat_transform,
        }
        self.dis_type = distance_type
        self.ori_sent = ori_sent

    def transform(self, sentence):
        self.ori_sent = sentence
        
        sents = []        
        # edit distance
        # transform_num = max(int(len(sentence)*self.degree), 1)
        if self.dis_type == "char":
            transform_num = math.ceil(len(sentence) * self.degree) 
        else:
            transform_num = math.ceil(len(sentence.split()) * self.degree) 

        for _ in range(self.aug_num):
            if self.dis_type == "char":
                indices = np.random.choice(len(sentence), len(sentence), replace=False)
            else:
                indices = np.random.choice(len(sentence.split()), len(sentence.split()), replace=False)
            # indices = np.random.choice(len(sentence), len(sentence), replace=False).tolist()
            # print(indices)
            sent = deepcopy(sentence)

            delta = 0
            for i, idx in enumerate(indices):
                if i >= transform_num:
                    break

                # idx += delta
                # random select one type of transformation
                transform_type = random.choice(self.transform_types)
                self.transformation = self.TRANSFORMATION[transform_type]
                if self.dis_type == "char":
                    char  = sent[idx]
                    try:
                        char = sent[idx]
                    except:
                        print(len(sent), idx)

                    if char == ' ' and transform_type != 'insert' and transform_type != 'repeat':
                        continue

                length_1 = len(sent) 
                if self.dis_type == "char":
                    candidate_sent = self.transformation(sent, idx)
                else:
                    candidate_sent = self.transform_target(sent,idx)
                if candidate_sent is None:
                    continue
                sent = candidate_sent
                length_2 = len(sent) 
                
                # update index mapping
                delta = length_2 - length_1 # 1 if length_2 < length_1 else -1
                if self.dis_type=='char':
                    for j, index in enumerate(indices):
                        if index >= idx:
                            indices[j] += delta

            if sent not in sents: 
                sents.append(sent)
        # if len(sents)>1:
        #     sents.remove(sentence)

        return sents


    def delete_transform(self, sent, i):
        if len(sent)==1:
            return None
        return sent[:i] + sent[i+1:]


    def insert_transform(self, sent, i):
        char = sent[i].lower()
        if char in keyboard_neighbors_keys:   
            keyboard_neighbor = random.choice(keyboard_neighbors[char])
            return sent[:i] + keyboard_neighbor + sent[i:]
        else:   
            return None
        


    def replace_transform(self, sent, i):
        char = sent[i].lower()
        if char in keyboard_neighbors_keys:
            keyboard_neighbor = random.choice(keyboard_neighbors[char])
            return sent[:i] + keyboard_neighbor + sent[i+1:]
        else:
            return None


    def swap_transform(self, sent, i):
        try:
            if sent[i-1] != ' ' and sent[i] != ' ':
                return sent[:i-1] + sent[i] + sent[i-1] + sent[i+1:]
            else:
                return None
        except:
            return None


    def repeat_transform(self, sent, i):
        char = sent[i]
        return sent[:i] + char + sent[i:]

    

    def transform_target(self, sent,sent_idx):
        # print(sent_idx)
        transform_type = random.choice(self.transform_types)
        self.trans = self.TRANSFORMATION[transform_type]
        sentence_tokens = sent.split()
        ori_sentence_tokens = self.ori_sent.split()
        # try:
        word = ori_sentence_tokens[sent_idx]
        # except:
        #     print(sentence_tokens)
        #     print(sent_idx)
        char_idx = np.random.choice(len(word), replace=False)
        substitute = self.trans(word,char_idx)
        if substitute == None:
            return None
        
        sentence_tokens[sent_idx] = substitute

        return ' '.join(sentence_tokens)




    def transform_char(self, sent,sent_idx):
        # print(sent_idx)
        transform_type = random.choice(self.transform_types)
        self.trans = self.TRANSFORMATION[transform_type]
        sentence_tokens = sent.split()
        # try:
        word = sentence_tokens[sent_idx]
        # except:
        #     print(sentence_tokens)
        #     print(sent_idx)
        char_idx = np.random.choice(len(word), replace=False)
        substitute = self.trans(word,char_idx)
        if substitute == None:
            return None
        
        sentence_tokens[sent_idx] = substitute

        return ' '.join(sentence_tokens)
    # def distance_measure(self, sentence1, sentence2):
    #     return distance.levenshtein(sentence1, sentence2)