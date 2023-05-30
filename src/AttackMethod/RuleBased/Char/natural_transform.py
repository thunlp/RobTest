from ..rule_transform import RuleTransform
from evaluator import EditDistance
import random
import numpy as np
from copy import deepcopy



class NaturalTransform(RuleTransform):
    def __init__(self, degree, aug_num=1):
        super().__init__(degree, aug_num)
        self.distance = EditDistance()
        self.natural_typos = {}
        for line in open('AttackMethod/RuleBased/Char/natural_transformation.txt'):
            line = line.strip().split()
            self.natural_typos[line[0]] = line[1:]
        self.natural_keys = self.natural_typos.keys()

    def transform(self, sentence):
        
        sents = []
        split = sentence.strip().split()
        sent_len = len(sentence.strip())

        for _ in range(self.aug_num):
            indices = np.random.choice(len(split), len(split), replace=False)
            word_split = deepcopy(split)
            for idx in indices:
                word = word_split[idx]
                word_split[idx] = self.natural_transform(word)
                if word_split[idx] == word:
                    continue
                if self.distance(" ".join(split), " ".join(word_split))/sent_len >= self.degree:
                    break 
            sents.append(" ".join(word_split))
        
        return sents

    def natural_transform(self, word):
        if word in self.natural_keys:
            return random.choice(self.natural_typos[word])
        else:
            return word