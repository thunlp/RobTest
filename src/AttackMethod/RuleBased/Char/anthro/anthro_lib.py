import numpy as np
import pandas as pd
import glob
import json
import pickle
import re
import Levenshtein
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import os
import enchant
from nltk.stem import PorterStemmer
import homoglyphs as hg
from spacy.lang.en import English


# English of the USA
english_checker = enchant.Dict("en_US") 

ps = PorterStemmer()
homoglyphs = hg.Homoglyphs(languages={'en'}, strategy=hg.STRATEGY_LOAD)


# check if the token is an English word
def is_english(token):
    if 'our' in token.lower() and english_checker.check(token.lower().replace('our','or')):
        return True
    if 'able' in token.lower() and english_checker.check(token.lower().replace('able','ible')):
        return True
    if 'ise' in token.lower() and english_checker.check(token.lower().replace('ise','ize')):
        return True
    if english_checker.check(token.lower()) or english_checker.check(ps.stem(token)):
        return True
    return False

def matchcodex(token, pre_suf_chars=3):
    token = re.sub('[1]+', 'l', token, flags=re.I)
    ## LEWU: Swap the order of the following two lines
    token = token.replace('*','').replace('+','t').replace('!','i')
    token = re.sub('[^0-9a-zA-Z]+', '', token, flags=re.I)
    token = token.upper()
    codex = []
    prev = None
    for i, c in enumerate(token):
        if c != " " and (i == 0 and c in "A@EI!O0UY") or (c not in "A@EI!O0UY" and c != prev):
            codex.append(c)
        prev = c
    if len(codex) > int(pre_suf_chars*2):
        return "".join(codex[:pre_suf_chars] + codex[-pre_suf_chars:])
    else:
        return "".join(codex)


def preprocess_onebyone(texts, batch_size=100):
    tokenizer = Tokenizer(English().vocab, token_match=re.compile(r'\S+').match)
    preproc_pipe = []
    for doc in tokenizer.pipe(texts, batch_size=batch_size):
        preproc_pipe.append(doc)
    return preproc_pipe

def good_pair_replacement(a, b):
    if a == b:
        return True
    a = a.replace('!','i').replace('+','t')
    b = b.replace('!','i').replace('+','t')

    pairs = {'iy', 'yi', '@a','a@','0o','o0','5s','s5','1l','l1','6b','b6','9g','g9','t+','+t'}
    if "{}{}".format(a,b) in pairs:
        return True
    return False


def good_pair_nearby(a, b):
    if a == b:
        return True
    pairs = {'iy', 'yi', 'ie', 'ye', '@a','a@','0o','o0', 'ya', 'uy', 'ue', 'ae'}
    if "{}{}".format(a,b) in pairs:
        return True
    return False


def check_vowel(a):
    VOWELS = 'a@eijo0uy!'
    try:
        a = homoglyphs.to_ascii(a)[0]
    except:
        return False # change this part that applies to both mfalse and mtrue
    if a in VOWELS:
        return True
    return False


def final_check(token, match):
    if is_english(match.lower()) and match.lower() != token.lower() and not is_english(token.lower()):
        return True

    token = token.lower()   
    match = match.lower()
    edits = Levenshtein.editops(token, match)

    for edit in edits:
        if edit[0] != 'delete':
            try:
                c = match[edit[2]]

                if edit[0] == 'replace':
                    try:
                        b_c = token[edit[1]-1]
                    except:
                        b_c = None

                    try:
                        a_c = token[edit[1]+1]
                    except:
                        a_c = None

                    o_c = token[edit[1]]

                    if check_vowel(c):
                        if not good_pair_replacement(o_c, c):
                            return True

                        if not check_vowel(o_c):
                            return True

                    else:
                        if c.isdigit() and not good_pair_replacement(o_c, c):
                            return True
                        if check_vowel(b_c) and check_vowel(a_c):
                            return True

                if edit[0] == 'insert' and check_vowel(c):
                    try:
                        b_c = token[edit[1]-1]
                    except:
                        b_c = None

                    try:
                        a_c = token[edit[1]]
                    except:
                        a_c = None

                    if b_c != None and not good_pair_nearby(b_c, c) and check_vowel(b_c):
                        return True

                    if a_c != None and not good_pair_nearby(c, a_c) and check_vowel(a_c):
                        return True

                    if b_c != None and a_c != None and check_vowel(b_c) and check_vowel(a_c) and c != b_c and c != a_c:
                        return True

                    if b_c != None and a_c != None and not check_vowel(b_c) and not check_vowel(a_c):
                        return True

            except Exception as e:
                print(e)
                print("ERROR!!! => EDIT", edit, token, match)

    return False
    

class ANTHRO:
    def __init__(self):
        self.max_level = 2
        self.token2idx = {}
        self.idx2token = {}
        self.level2code2token = {}
        self.token2spell = {}
        self.__version__ = 'Public'
        self.preserve_last_threshold = 3

        for level in range(self.max_level):
            self.level2code2token[str(level)] = {}


    def save(self, path, data=None):
        from json import JSONEncoder
        class set_encoder(JSONEncoder):
            def default(self, obj):
                return list(obj)
                
        if not data:
            data = {
                'token2idx': self.token2idx,
                'level2code2token': self.level2code2token,
                'token2spell': self.token2spell,
            }

        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError

        try:
            os.makedirs("{}/".format(path))
        except:
            pass

        for k in data:
            print("saving...", k)
            with open("{}/{}.json".format(path, k), 'w', encoding='utf-8') as f:
                json.dump(data[k].items(), f, cls=set_encoder)

    def load(self, path):
        keys = {
            'token2idx': 'token2idx',
            'level2code2token': 'level2code2token',
            'token2spell': 'token2spell'
        }

        data = {}
        for k in keys:
            # print("loading...", k)
            with open("{}/{}.json".format(path, keys[k]), 'r', encoding='utf-8') as f:
                data[k] = json.load(f)
                setattr(self, k, dict(data[k]))

        # print("loaded")
        # self.statistics()

        for level in self.level2code2token:
            codes = self.level2code2token[str(level)]
            for code in codes:
                codes[code] = set(codes[code])

        self.convert_token_idx()
        self.fix_token2spell()

        # print("done")


    def fix_token2spell(self):
        # for token in tqdm(self.token2spell.keys()):
        for token in self.token2spell.keys():
            if 'list' in str(type(self.token2spell[token])) and len(self.token2spell[token]) > 0:
                if 'list' in str(type(self.token2spell[token][0])):
                        self.token2spell[token] = set(list(np.concatenate(self.token2spell[token])))


    def add_raw_tokens(self, raw_tokens, level):
        codes = self.level2code2token[str(level)]
        for token in tqdm(raw_tokens):
            token = str(token)
            token_lower = token.lower()
            
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
            if token_lower not in self.token2idx:
                self.token2idx[token_lower] = len(self.token2idx)
            code = self.soundex(token_lower, k=level)

            if code not in codes:
                codes[code] = set()
            if self.token2idx[token_lower] not in codes[code]:
                codes[code].add(self.token2idx[token_lower])

            if self.token2idx[token_lower] not in self.token2spell:
                self.token2spell[self.token2idx[token_lower]] = set()
            self.token2spell[self.token2idx[token_lower]].add(self.token2idx[token])

    def process(self, texts, raw_tokens=None):
        if not raw_tokens:
            sentence_tokens = preprocess_onebyone(texts)

            raw_tokens = []
            for j, tokens in tqdm(enumerate(sentence_tokens)):
                tokens = list(tokens)
                raw_tokens.extend(tokens)

        for level in range(self.max_level):
            if level > 0:
                print("...level", level)
                self.add_raw_tokens(raw_tokens, level)

        self.convert_token_idx()

    def convert_token_idx(self):
        # for token in tqdm(self.token2idx):
        for token in self.token2idx:
            self.idx2token[int(self.token2idx[token])] = token

    def statistics(self):
        print("UPPD VERSION", self.__version__)
        print("Total unique codes:")
        for level in self.level2code2token:
            codes = self.level2code2token[str(level)]
            print("Level {}: {} codes".format(level, len(codes)))

    def soundex(self, token, k=0):
        soundex_result = token[:k+1].upper()
        token = re.sub('[hw]', '', token, flags=re.I)
        token = re.sub('[-]','', token, flags=re.I)
        token = re.sub('[l1]+', '4', token, flags=re.I)
        token = re.sub('[bfpv]+', '1', token, flags=re.I)
        token = re.sub('[cgjkqsxz]+', '2', token, flags=re.I)
        token = re.sub('[dt]+', '3', token, flags=re.I)
        token = re.sub('[mn]+', '5', token, flags=re.I)
        token = re.sub('r+', '6', token, flags=re.I)
        token = token[k+1:]
        token = re.sub('[a@eio0uy]','', token, flags=re.I)
        soundex_result += token[0:3+k]
        if len(soundex_result) < 4+k:
            soundex_result += '0'*(4+k-len(soundex_result))
        return soundex_result

    def get_similars(self, 
                    token, 
                    level=0, 
                    distance=2, 
                    min_chars=3,
                    strict=True):

        def preprocess(token):
            return token

        if len(token) < min_chars:
            return []

        code = self.soundex(token, k=int(level))
        soundex_codes = self.level2code2token[str(level)]

        if code not in soundex_codes:
            return []

        cands = soundex_codes[code]
        cands = [self.idx2token[cand] for cand in cands]

        if strict:
            matchcode_token = matchcodex(token)
            cands = [cand for cand in cands if matchcodex(cand) == matchcode_token]

        dists = np.array([Levenshtein.distance(token.lower(), cand.lower()) for cand in cands])
        idx = np.where(dists <= distance)[0]

        cands = [list(cands)[i] for i in idx]
        cands = set([cand.lower() for cand in cands])

        rt = []
        for cand in cands:
            rt += [self.idx2token[a] for a in list(self.token2spell[self.token2idx[cand]])]

        rt = set([preprocess(a) for a in rt \
            if (good_pair_replacement(a[-1].lower(),token[-1].lower()) and len(a) > self.preserve_last_threshold) or 
            (len(a) <= self.preserve_last_threshold and a[-1].lower() == token[-1].lower())])

        if strict:
            rt = set(a for a in rt if not final_check(token.lower(), a.lower()))

        return rt