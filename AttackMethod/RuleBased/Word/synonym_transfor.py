from ..rule_transform import RuleTransform
from transformers import pipeline
from copy import deepcopy
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from pyinflect import getInflection
import OpenHowNet
import os
import numpy as np
import random
import torch
import math
import atexit
# OpenHowNet.download()
from tqdm import tqdm




os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# before applying synonym transform, you may need to download the nltk corpus
# import nltk
# nltk.download('wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'punkt')



path_to_jar = '/data/private/gaohongcheng/BenchmarkRobustness-NLP-main/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
path_to_model = '/data/private/gaohongcheng/BenchmarkRobustness-NLP-main/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
stop_words = {'!', '"', '#', '$', '%', '&', "'", "'s", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
              '?', '@', '[', '\\', ']', '^', '_', '`', '``', 'a', 'about', 'above', 'after', 'again', 'against', 'ain',
              'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before',
              'being', 'below', 'between', 'both', 'but', 'by', 'ca', 'can', 'couldn', "couldn't", 'd', 'did', 'didn',
              "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few',
              'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't",
              'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
              'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't",
              'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'n\'t',
              'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
              're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so',
              'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
              'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'us', 've',
              'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which',
              'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd",
              "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '{', '|', '}', '~'}
sememe_map = {
    'noun': wn.NOUN,
    'verb': wn.VERB,
    'adj': wn.ADJ,
    'adv': wn.ADV,
    'num': 0,
    'letter': 0,
    'pp': wn.NOUN,
    'pun': 0,
    'conj': 0,
    'echo': 0,
    'prep': 0,
    'pron': 0,
    'wh': 0,
    'infs': 0,
    'aux': 0,
    'expr': 0,
    'root': 0,
    'coor': 0,
    'prefix': 0,
    'det': 0,
}

wordnet_map = {
    "N": wn.NOUN,
    "V": wn.VERB,
    "J": wn.ADJ,
    "R": wn.ADV,
    'n': wn.NOUN,
    'v': wn.VERB,
    'j': wn.ADJ,
    'r': wn.ADV
}

# print('in synonym:', 0)
synonym_dict = {}
antonym_dict = {}
synonym_dict_path = os.path.join(os.path.dirname(__file__), 'synonym_dict.pt')
antonym_dict_path = os.path.join(os.path.dirname(__file__), 'antonym_dict.pt')
# sent_dict_path = os.path.join(os.path.dirname(__file__), 'syn_sent_dict.pt')


def init_wordnet():
    global synonym_dict, antonym_dict, synonym_dict_path, antonym_dict_path
    synonym_dict = torch.load(synonym_dict_path) if os.path.exists(synonym_dict_path) else {}
    antonym_dict = torch.load(antonym_dict_path) if os.path.exists(antonym_dict_path) else {}
    # print(synonym_dict, antonym_dict)



@atexit.register
def save_dict():
    if len(synonym_dict.keys()) > 0:
        torch.save(synonym_dict, synonym_dict_path)
    if len(antonym_dict.keys()) > 0:
        torch.save(antonym_dict, antonym_dict_path)

# print('in synonym:', 1)

class SynonymTransform(RuleTransform):
    
    def __init__(self, degree, aug_num=1,ori_sent=''):
        super().__init__(degree, aug_num)

        self.mlm_ckpt='bert-base-uncased'
        self.pos_tagger = StanfordPOSTagger(model_filename=path_to_model, path_to_jar=path_to_jar)
        self.ltz = WordNetLemmatizer()
        self.transform_types = ['wordnet', 'hownet']
        self.TRANSFORMATION = {
            'wordnet': self.wordnet_substitute, 
            'hownet': self.hownet_substitute, 
        }
        
        init_wordnet()
        self.ori_sent = ori_sent
        self.sememe_dict = OpenHowNet.HowNetDict()
        # self.sent_dict = torch.load(sent_dict_path) if os.path.exists(sent_dict_path) else {}
        if len(ori_sent)>1:
            self.sents = [ori_sent]
            self.sent_tag = self.pos_tag_wordnet(ori_sent.split())
            self.wordnet_dict = self.get_wordnet_dict(ori_sent)
            self.hownet_dict = self.get_hownet_dict(ori_sent)
                # self.sent_dict[self.ori_sent] = [self.wordnet_dict,self.hownet_dict]
                # torch.save(self.sent_dict, sent_dict_path)
                



    def transform(self, sentence): 
        self.ori_sent = sentence.strip()      

        self.sent_tag = self.pos_tag_wordnet(sentence.split())
        self.wordnet_dict = self.get_wordnet_dict(sentence)
        self.hownet_dict = self.get_hownet_dict(sentence)


        self.word_split = self.ori_sent.split()
        # substitute_num = max(int(len(word_split)*self.degree), 1)
        transform_num = math.ceil(len(self.word_split) * self.degree)
        self.sents = [sentence]
        self.splits = [sentence.split()]
        for _ in range(self.aug_num):
            count = 0
            indices = np.random.choice(len(self.word_split), len(self.word_split), replace=False)
            candidate_split = self.word_split
            # iteratively substitute words in sentence, the iteration number is determined by self.degree
            for idx in indices:
                if count >= transform_num:
                    break
                transform_type = random.choice(self.transform_types)
                self.substitute = self.TRANSFORMATION[transform_type]

                candidate_split = self.substitute(candidate_split, idx)

                if candidate_split is None:
                    candidate_split = self.splits[-1]
                    continue
                count += 1

            if ' '.join(candidate_split)  not in self.sents:
                self.sents.append(' '.join(candidate_split))
                self.splits.append(candidate_split)
        if len(self.sents)>1:
            self.sents.remove(sentence)
        # print(len(self.sents))
        return self.sents



    def wordnet_substitute(self, split, idx):   
        synonyms = self.wordnet_dict[idx]
        mask_word = self.word_split()[idx]

        candidate_splits = []
        for synonym in synonyms:
            word_split = deepcopy(split)
            pred_word = synonym
            word_split[idx] = synonym
            candidate_split = word_split
            # candidate_sent = ' '.join(word_split)      
            # deduplicate
            if ' '.join(candidate_split).lower() == ' '.join(self.word_split).lower() or ' '.join(candidate_split) in self.sents:
                continue

            is_antonym = self.check_antonyms(mask_word, pred_word)

            if is_antonym:
                continue
            candidate_splits.append(candidate_split)

        if len(candidate_splits) >= 1:
            return random.choice(candidate_splits)
        else:
            return None

    def hownet_substitute(self, split, idx):
        '''Gets a list of candidates for each word using sememe.
        '''
        filtered_replacements =self.hownet_dict[idx]
        mask_word= self.word_split()[idx]

        candidate_splits = []
        for pred_word in filtered_replacements:
            # check anonyms 
            is_antonym = self.check_antonyms(mask_word, pred_word)
            if is_antonym:
                continue
            word_split = deepcopy(split)
            word_split[idx] = pred_word
            # candidate_sent = ' '.join(word_split)
            candidate_split = word_split
            if ' '.join(candidate_split) not in self.sents:
                candidate_splits.append(candidate_split)
        if len(candidate_splits) >= 1:
            return random.choice(candidate_splits)
        else: 
            return None
        

    def get_hownet_dict(self,sentence):
        self.total_replacements = {}
        total_filtered_reps = []
        # total_filtered_reps = []
        # words = [orig_words[x] for x in range(len(orig_words))]
        words = sentence.split()
        tags = self.sent_tag
        
        for i, word in enumerate(words):
            # print(word)
            filtered_replacements = []
            word = self.ltz.lemmatize(word, tags[i][1])

            replacements = self.memonized_get_replacements(word, self.sememe_dict)

            for candidate_tuple in replacements:
                [candidate, pos] = candidate_tuple
                try:
                    # print(sememe_map[pos])
                    if ((not candidate == '') and
                        (not candidate == word) and  # must be different
                            (sememe_map[pos] == tags[i][1]) and  # part of speech tag must match
                            (candidate not in stop_words)):
                        infl = getInflection(candidate, tag=tags[i][2], inflect_oov=True)
                        if infl and infl[0]:
                            filtered_replacements.append(infl[0])
                        else:
                            filtered_replacements.append(candidate)
                except:
                    continue

            filtered_replacements = list(set(filtered_replacements))
            print(filtered_replacements)
            total_filtered_reps.append(filtered_replacements)

        return total_filtered_reps

    def get_wordnet_dict(self,sentence):
        total_synonyms = []
        split = sentence.split()
        # substitute with synonyms
        
        # check pos tag
        for i in range(len(split)):
            mask_word = split[i]
            wordnet_tag = self.sent_tag[i][1]
            # word = self.ltz.lemmatize(mask_word, wordnet_tag)
            synonyms = self.get_synonyms(mask_word, wordnet_tag)
            total_synonyms.append(synonyms)
        return total_synonyms



    def check_pos_tag(self, sent, idx): # return bool
        pos_tags = self.pos_tagger.tag(sent.split()) 
        return pos_tags[idx][1]

    
    def get_synonyms(self, mask_word, tag):
        
        if f'{mask_word}.{tag}' in synonym_dict.keys():
            # print('get synonym from cache')
            return synonym_dict[f'{mask_word}.{tag}']
        
        # print('get synonym online')
        synonyms = set()

        for syn in wn.synsets(mask_word):
            if syn.name().split('.')[1] != tag: # should be the same pos tag
                continue
            for lemma in syn.lemmas():   
                synonyms.add(lemma.name())
        if mask_word in synonyms:
            synonyms.remove(mask_word)

        synonyms = list(synonyms)
        synonym_dict[f'{mask_word}.{tag}'] = synonyms

        return synonyms


    def get_word_antonyms(self, word):
        
        if word in antonym_dict.keys():
            # print('get antonym from cache')
            return antonym_dict[word]

        try:
            # print('get antonym online')
            antonyms_lists = set()

            for syn in wn.synsets(word):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms_lists.add(l.antonyms()[0].name())
            antonyms_lists = list(antonyms_lists)
            antonym_dict[word] = antonyms_lists
            return antonyms_lists
        except:
            antonym_dict[word] = []
            return []


    def pos_tag_wordnet(self, text):
        """
            Create pos_tag with wordnet format
        """
        pos_tagged_text = nltk.pos_tag(text)
        pos_tags = self.pos_tagger.tag(text)

        # map the pos tagging output with wordnet output
        pos_tagged_text = [
            (pos_tagged_text[i][0], wordnet_map.get(pos_tagged_text[i][1][0]), pos_tags[i][1]) if pos_tagged_text[i][1][
                                                                                                    0] in wordnet_map.keys()
            else (pos_tagged_text[i][0], wn.NOUN, pos_tags[i][1])
            for i in range(len(pos_tagged_text))
        ]
        # print('pos_tagged_text', pos_tagged_text)
        return pos_tagged_text


    def memonized_get_replacements(self, word, sememe_dict):
        if word in self.total_replacements:
            pass
        else:
            word_replacements = []
            # Get candidates using sememe from word

            sememe_tree = sememe_dict.get_sememes_by_word(word, structured=True, lang="en", merge=False)

            for sense in sememe_tree:
                # For each sense, look up all the replacements
                synonyms = sense['word']['syn']
                for synonym in synonyms:
                    actual_word = sememe_dict.get(synonym['id'])[0]['en_word']
                    actual_pos = sememe_dict.get(synonym['id'])[0]['en_grammar']
                    word_replacements.append([actual_word, actual_pos])
            self.total_replacements[word] = word_replacements
        return self.total_replacements[word]


    def check_antonyms(self, mask_word, pred_word):
        # filter antonyms
        antonyms_list = self.get_word_antonyms(mask_word)
        return True if pred_word in antonyms_list else False


    

    def transform_target(self, sent,sent_idx):
        transform_type = random.choice(self.transform_types)
        self.trans = self.TRANSFORMATION[transform_type]
        # sentence_tokens = sent.split()
        # ori_sentence_tokens = self.ori_sent.split()
        # sentence_tokens[sent_idx] = ori_sentence_tokens[sent_idx]
        return self.trans(sent.strip(),sent_idx)