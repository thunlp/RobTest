from ..rule_transform import RuleTransform
# import distance
import random
from copy import deepcopy
import numpy as np
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from .perturbations_store import PerturbationsStorage
from gensim.models import KeyedVectors as W2Vec               
model = W2Vec.load_word2vec_format("src/naacl2019-like-humans-visual-attacks/VIPER/vce.normalized")
perturbations_file='src/perturbations.txt'
perturbations_file = PerturbationsStorage(perturbations_file)
topn = 20


descs = pd.read_csv('src/data/NamesList.txt', skiprows=np.arange(16), header=None, names=['code', 'description'])
descs = descs.dropna(0)
descs_arr = descs.values # remove the rows after the descriptions
vectorizer = CountVectorizer(max_features=1000)
desc_vecs = vectorizer.fit_transform(descs_arr[:, 0]).astype(float)
vecsize = desc_vecs.shape[1]
vec_colnames = np.arange(vecsize)
desc_vecs = pd.DataFrame(desc_vecs.todense(), index=descs.index, columns=vec_colnames)
descs = pd.concat([descs, desc_vecs], axis=1)

def char_to_hex_string(ch):
    return '{:04x}'.format(ord(ch)).upper()


disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
              'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
              'CANADIAN', 'YI', 'LYCIAN', 'HANGUL', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
              'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN', 
              'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
              'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
              'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
              'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN'
             ]

disallowed_codes = ['1F1A4', 'A7AF']




# function for retrieving the variations of a character
def get_all_variations(ch):
       
    # get unicode number for c
    c = char_to_hex_string(ch)
    
    # problem: latin small characters seem to be missing?
    if np.any(descs['code'] == c):
        description = descs['description'][descs['code'] == c].values[0]
    else:
        print('Failed to disturb %s, with code %s' % (ch, c))
        return c, np.array([])
    
    # strip away everything that is generic wording, e.g. all words with > 1 character in
    toks = description.split(' ')

    case = 'unknown'

    identifiers = []
    for tok in toks:
           
        if len(tok) == 1:
            identifiers.append(tok)
            
            # for debugging 
            if len(identifiers) > 1:
                print('Found multiple ids: ')
                print(identifiers)

        elif tok == 'SMALL':
            case = 'SMALL'
        elif tok == 'CAPITAL':
            case = 'CAPITAL'

    # for debugging
    #if case == 'unknown':
    #    sys.stderr.write('Unknown case:')
    #    sys.stderr.write("{}\n".format(toks))

    # find matching chars
    matches = []
    
    for i in identifiers:        
        for idx in descs.index:
            desc_toks = descs['description'][idx].split(' ')
            if i in desc_toks and not np.any(np.in1d(desc_toks, disallowed)) and \
                    not np.any(np.in1d(descs['code'][idx], disallowed_codes)) and \
                    not int(descs['code'][idx], 16) > 30000:

                # get the first case descriptor in the description
                desc_toks = np.array(desc_toks)
                case_descriptor = desc_toks[ (desc_toks == 'SMALL') | (desc_toks == 'CAPITAL') ]

                if len(case_descriptor) > 1:
                    case_descriptor = case_descriptor[0]
                elif len(case_descriptor) == 0:
                    case = 'unknown'

                if case == 'unknown' or case == case_descriptor:
                    matches.append(idx)

    # check the capitalisation of the chars
    return c, np.array(matches)

# function for finding the nearest neighbours of a given word
def get_unicode_desc_nn(c, perturbations_file, topn=1):
    # we need to consider only variations of the same letter -- get those first, then apply NN
    c, matches = get_all_variations(c)
    
    if not len(matches):
        return [], [] # cannot disturb this one
    
    # get their description vectors
    match_vecs = descs[vec_colnames].loc[matches]
           
    # find nearest neighbours
    neigh = NearestNeighbors(metric='euclidean')
    Y = match_vecs.values
    neigh.fit(Y) 
    
    X = descs[vec_colnames].values[descs['code'] == c]

    if Y.shape[0] > topn:
        dists, idxs = neigh.kneighbors(X, topn, return_distance=True)
    else:
        dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

    # turn distances to some heuristic probabilities
    #print(dists.flatten())
    probs = np.exp(-0.5 * dists.flatten())
    probs = probs / np.sum(probs)
    
    # turn idxs back to chars
    #print(idxs.flatten())
    charcodes = descs['code'][matches[idxs.flatten()]]
    
    #print(charcodes.values.flatten())
    
    chars = []
    for charcode in charcodes:
        chars.append(chr(int(charcode, 16)))

    # filter chars to ensure OOV scenario (if perturbations file from prev. perturbation contains any data...)
    c_orig = chr(int(c, 16))
    chars = [char for char in chars if not perturbations_file.observed(c_orig, char)]

    #print(chars)

    return chars, probs

def readD(fn):
  h = {}
  for line in open(fn):
    line = line.strip()
    x = line.split()
    a,b = x[0].strip(),x[1].strip()
    h[a] = b
  return h
# copy from textbugger
similar_chars = {
    'i':['і', '1'], 'l':['ⅼ', '1'], 'z':['ᴢ', '2'],   "s":['5', 'ѕ'], "g":['ɡ','9'], 'b':['Ь','6'], 'q':['ԛ', '9'], 'o':['0','о' ], '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
         "'": '`', 'a': 'ɑ',  'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏',  'h': 'հ', 'j': 'ϳ',
         'k': '𝒌',  'm': 'ｍ', 'n': 'ո', 'p': 'р',  'r': 'ⲅ',  't': '𝚝', 'u': 'ս',
         'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у'
}
similar_chars_keys = similar_chars.keys()
# h = readD("/data/private/gaohongcheng/naacl2019-like-humans-visual-attacks-master/code/VIPER/selected.neighbors")

class GlyphTransform(RuleTransform):
    def __init__(self, degree, aug_num=1,distance_type="char",ori_sent=''):
        # print(distance_type)
        super().__init__(degree, aug_num)
        self.dis_type = distance_type
        # print(self.distance_type)
        self.ori_sent = ori_sent
        self.mydict_dces= {}
        self.mydict_ices={}
        self.h = readD("./naacl2019-like-humans-visual-attacks/VIPER/selected.neighbors")
        self.transform_types = ['similar','dces','eces','ices']
        # self.transform_types = ['wordnet']
        self.TRANSFORMATION = {
            'similar': self.get_similar, 
            'dces': self.get_dces,
            'eces': self.get_eces,
            'ices':self.get_ices
        }


    def transform(self, sentence):
        self.ori_sent = sentence
        sents = []
        # edit distance = int(len(split)*self.degree)
        # transform_num = max(int(len(sentence.strip())*self.degree), 1)
        # transform_num = math.ceil(len(sentence) * self.degree)

        if self.dis_type == "char":
            transform_num = math.ceil(len(sentence) * self.degree) 
        else:
            transform_num = math.ceil(len(sentence.split()) * self.degree) 


        for _ in range(self.aug_num):
            transform_type = random.choice(self.transform_types)
            self.trans = self.TRANSFORMATION[transform_type]
            if self.dis_type == "char":
                indices = np.random.choice(len(sentence), len(sentence), replace=False)
            else:
                indices = np.random.choice(len(sentence.split()), len(sentence.split()), replace=False)
            count = 0
            sent = deepcopy(sentence)
            for idx in indices:
                if count >= transform_num:
                    break
                if self.dis_type == "char":
                    char  = sent[idx]
               

                    if char ==' ':  
                        continue
                    substitute = self.trans(char)

                    if char == substitute:
                        continue
                    if type(substitute) == str:
                        sent = deepcopy(sent[:idx] +substitute + sent[idx+1:])
                    else:
                        sent = deepcopy(sent[:idx] + substitute[np.random.choice(len(substitute))] + sent[idx+1:])
                else:
                    candidate_sent = self.transform_target(sent,idx)
                    if candidate_sent  == None:
                        continue
                    sent = candidate_sent 
                    
                count += 1
            if sent not in sents:
                sents.append(sent)
        # if len(sents)>1:
        #     sents.remove(sentence)
        return sents

    def transform_target(self, sent,sent_idx):
        # print("---------------")
        # print(sent)
        transform_type = random.choice(self.transform_types)
        self.trans = self.TRANSFORMATION[transform_type]
        sentence_tokens = sent.split()
        ori_sentence_tokens = self.ori_sent.split()
        try:
            word = ori_sentence_tokens[sent_idx]
        except:
            print(ori_sentence_tokens)
            print(sentence_tokens)
            print(sent_idx)
        word = ori_sentence_tokens[sent_idx]
        idx = np.random.choice(len(word), len(word), replace=False)[0]
        char = word[idx]
        substitute = self.trans(char)

        if char == substitute or substitute==None:
            return None        
        if type(substitute) == str:
            word = deepcopy(word[:idx] + substitute + word[idx+1:])
        else:
            word = deepcopy(word[:idx] + substitute[np.random.choice(len(substitute))] + word[idx+1:])
        sentence_tokens[sent_idx] = word
        # if len(sentence_tokens) == len(ori_sentence_tokens):
        return ' '.join(sentence_tokens)
        # else:
        #     print("fuck")
        #     return None

    def transform_char(self, sent,sent_idx):
        transform_type = random.choice(self.transform_types)
        self.trans = self.TRANSFORMATION[transform_type]
        sentence_tokens = sent.split()
        word = sentence_tokens[sent_idx]
        idx = np.random.choice(len(word), len(word), replace=False)[0]
        char = word[idx]
        substitute = self.trans(char)

        if char == substitute or substitute==None:
            return None        
        if type(substitute) == str:
            word = deepcopy(word[:idx] + substitute + word[idx+1:])
        else:
            word = deepcopy(word[:idx] + substitute[np.random.choice(len(substitute))] + word[idx+1:])
        sentence_tokens[sent_idx] = word
        return ' '.join(sentence_tokens)



        # def distance_measure(self, sentence1, sentence2):
        #     return distance.levenshtein(sentence1, sentence2)
    def get_ices(self,c):
        if c not in self.mydict_ices:
            try:
                similar = model.most_similar(c, topn=topn)
            except:
                return c
            # print(similar)
            words, probs = [x[0] for x in similar], np.array([x[1] for x in similar])
            probs /= np.sum(probs)
            self.mydict_ices[c] = (words, probs)
        else:
            words, probs = self.mydict_ices[c]

        s = np.random.choice(words, 1, replace=True, p=probs)[0]
        if words==[]:
            return c
        return s

    # def get_ices()


    def get_dces(self,c):
        if c not in self.mydict_dces:
            similar_chars, probs = get_unicode_desc_nn(c, perturbations_file, topn=topn)
            probs = probs[:len(similar_chars)]

            # normalise after selecting a subset of similar characters
            probs = probs / np.sum(probs)

            self.mydict_dces[c] = (similar_chars, probs)

        else:
            similar_chars, probs = self.mydict_dces[c]
        
        if similar_chars == []:
            return c
        s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
        return s


    def get_eces(self,c):
        r = self.h.get(c,c)
        if len(r)>0:
            return r
        return None

    def get_similar(self,c):
        if c in similar_chars_keys:
            return similar_chars[c]
        return c




