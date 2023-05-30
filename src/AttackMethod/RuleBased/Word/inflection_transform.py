from ..rule_transform import RuleTransform
from transformers import pipeline
from copy import deepcopy

import numpy as np
import random

import math

# OpenHowNet.download()


from typing import Dict
# import random, nltk, lemminflect
import nltk,lemminflect

class InflectionTransform(RuleTransform):
    
    def __init__(self, degree, aug_num=1,ori_sent=''):
        super().__init__(degree, aug_num)

        # self.mlm_ckpt='bert-base-uncased'
        # self.pos_tagger = StanfordPOSTagger(model_filename=path_to_model, path_to_jar=path_to_jar)
        # self.ltz = WordNetLemmatizer()
        # self.transform_types = ['wordnet', 'hownet']
        # self.TRANSFORMATION = {
        #     'wordnet': self.wordnet_substitute, 
        #     'hownet': self.hownet_substitute, 
        # }
        
        # init_wordnet()
        # self.degree = degree
        self.ori_sent = ori_sent
        self.sent_tag = nltk.pos_tag(self.ori_sent.split(),tagset='universal') 

        # self.sememe_dict = OpenHowNet.HowNetDict()
        # self.sent_dict = sent_dict
        # if len(ori_sent)>1:
        #     self.sents = [ori_sent]
        #     if ' '.join(self.ori_sent.split()[0:15]+self.ori_sent.split()[-5:]) in self.sent_dict:
        #         self.wordnet_dict,self.hownet_dict = self.sent_dict[' '.join(self.ori_sent.split()[0:15]+self.ori_sent.split()[-5:])][0],self.sent_dict[' '.join(self.ori_sent.split()[0:15]+self.ori_sent.split()[-5:])][1]
            # else:
            #     self.sent_tag = self.pos_tag_wordnet(ori_sent.split())
            #     self.wordnet_dict = self.get_wordnet_dict(ori_sent)
            #     self.hownet_dict = self.get_hownet_dict(ori_sent)
            #     self.sent_dict[self.ori_sent] = [self.wordnet_dict,self.hownet_dict]
                # torch.save(self.sent_dict, sent_dict_path)
                



    def transform(self, sentence): 
        self.ori_sent = sentence      
        self.sent_tag = nltk.pos_tag(sentence.split(),tagset='universal') 


        word_split = sentence.split()
        transform_num = math.ceil(len(word_split) * self.degree)
        self.sents = [sentence]
        for _ in range(self.aug_num):
            count = 0
            indices = np.random.choice(len(word_split), len(word_split), replace=False)
            candidate_sent = deepcopy(sentence)
            # iteratively substitute words in sentence, the iteration number is determined by self.degree
            # print(indices)
            for idx in indices:
                if count >= transform_num:
                    break
                # transform_type = random.choice(self.transform_types)
                candidate_sent  = self.random_inflect(candidate_sent.strip(), idx)


                if candidate_sent is None:
                    candidate_sent = self.sents[-1]
                    continue
                if candidate_sent == self.ori_sent:
                    continue
                count += 1

            if candidate_sent not in self.sents:
                self.sents.append(candidate_sent)
        if len(self.sents)>1:
            self.sents.remove(sentence)
        # print(len(self.sents))
        return self.sents

    def random_inflect(self,source: str,i,inflection_counts: Dict[str,int]=None) -> str:
        have_inflections = {'NOUN', 'VERB', 'ADJ'}
        # tokenized = MosesTokenizer(lang='en').tokenize(source) # Tokenize the sentence
        tokenized = source.split()

        upper = False
        if tokenized[0][0].isupper():
            upper = True
            tokenized[0]= tokenized[0].lower()
        
        pos_tagged =self.sent_tag # POS tag words in sentence

        word = self.ori_sent.split()[i]
        
        
        lemmas = lemminflect.getAllLemmas(word)
        # Only operate on content words (nouns/verbs/adjectives)
        if lemmas and pos_tagged[i][1] in have_inflections and pos_tagged[i][1] in lemmas:
            lemma = lemmas[pos_tagged[i][1]][0]
            inflections = (i, [(tag, infl) 
                            for tag, tup in 
                            lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).items() 
                            for infl in tup])
            if inflections[1]:
                # Use inflection distribution for weighted random sampling if specified
                # Otherwise unweighted
                if inflection_counts:
                    counts = [inflection_counts[tag] for tag, infl in inflections[1]]
                    inflection = random.choices(inflections[1], weights=counts)[0][1]
                else:
                    inflection = random.choices(inflections[1])[0][1]
                tokenized[i] = inflection
        if upper:
            tokenized[0] = tokenized[0].title()
        # return MosesDetokenizer(lang='en').detokenize(tokenized)
        return " ".join(tokenized)



    

    def transform_target(self, sent,sent_idx):
        # transform_type = random.choice(self.transform_types)
        # self.trans = self.TRANSFORMATION[transform_type]
        # sentence_tokens = sent.split()
        # ori_sentence_tokens = self.ori_sent.split()
        # sentence_tokens[sent_idx] = ori_sentence_tokens[sent_idx]
        return self.random_inflect(sent.strip(),sent_idx)