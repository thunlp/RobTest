from cmath import inf
from ..rule_transform import RuleTransform
import re
import random
from copy import deepcopy
from evaluator import USE, SentenceEncoder
import torch


def load_random_sentences_from_wikitext103(k):
# https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
    sentences = []
    with open("./data/wikitext-103/wiki.valid.tokens") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences += line.split('.')
    random.shuffle(sentences)
    return sentences[0:k]
z = load_random_sentences_from_wikitext103(100)

use = SentenceEncoder()

class DistractionTransform(RuleTransform):
    

    def __init__(self, degree, aug_num=1,dataset="",k=100):
        super().__init__(degree, aug_num)
        self.transform_types = ['url', 'statement','wiki','random'] 
        self.TRANSFORMATION = {
            'url': self.url_transform, 
            'statement': self.addition_transform, 
            'wiki':self.wiki_transform,
            'random':self.random_transform
        }
        self.token = self.dataset_token(dataset)
        self.additions =  ['and true is true', 'and false is not true']
        self.use = use
        self.wiki = z
 

    def transform(self, sentence):       
        sents = []        
        self.use_thred = 1 - self.degree
        max_trial = 10
        for _ in range(self.aug_num):
            sent = deepcopy(sentence)
            candidate_sent = deepcopy(sentence)
            
            for _ in range(max_trial):
                use_score = self.use.get_sim(sentence, candidate_sent)
                if use_score > self.use_thred:
                    sent = candidate_sent
                    # print(candidate_sent, use_score)
                else:
                    break
                 # random select one type of transformation
                transform_type = random.choice(self.transform_types)
                self.transformation = self.TRANSFORMATION[transform_type]
                candidate_sent = self.transformation(sent)
                
            
            
            if sent not in sents:
                sents.append(sent)
        return sents


    def dataset_token(self, dataset):
        token = []
        d = dataset[0:30]
        for sent,label in d:
            token.extend(sent.split())

        return token
    
    def addition_transform(self, sent):
        addition = random.choice(self.additions)
        return self.distraction(sent, addition)


    def url_transform(self, sent):
        url = ['https:/']
        split_sum = random.randint(1, 5)
        for _ in range(split_sum):
            length = random.randint(1, 5)
            url.append(self.generate_random_str(length))
        url = '/'.join(url)
        return self.distraction(sent, url)

    # def wiki_transform():
    #     with open(config.WIKITEXT103_DEV_PATH) as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             sentences += line.split('.')
    #     random.shuffle(sentences)
    #     return sentences[0]


    def distraction(self, sent, addition):
        sent_parts = re.compile("([.!?][\s]*)").split(sent)
        # print(sent_parts)
        if len(sent_parts) == 1:
            sent = sent + addition
            # print(sent, addition)
            # print(sent)
        else:
            sent = sent_parts[0] + addition + ' ' + sent_parts[1]
            # print(sent_parts[0], addition, sent_parts[1])
            # print(sent)
        return sent


    def generate_random_str(self, randomlength):
        random_str =''
        base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789.?/'
        length =len(base_str) -1
        for _ in range(randomlength):
            random_str +=base_str[random.randint(0, length)]
        return random_str




    def load_random_sentences_from_wikitext103(self,k):
    # https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
        sentences = []
        with open("/data/private/gaohongcheng/BenchmarkRobustness-NLP-main/data/wikitext-103/wiki.valid.tokens") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sentences += line.split('.')
        random.shuffle(sentences)
        return sentences[0:k]

    def wiki_transform(self, sent):
        wiki_distr = random.choice(self.wiki)
        return self.distraction(sent, wiki_distr)

    def random_transform(self, sent):
        split_sum = random.randint(1, 5)
        token = []
        for _ in range(split_sum):
            token.append(random.choice(self.token))
        tokens = ' '.join(token)
        return self.distraction(sent, tokens)
    
