from copy import deepcopy
import os

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os
# from evaluator import GPT2LM, GrammarChecker, USE
from AttackMethod.RuleBased import load_rule_transformer
from tqdm import tqdm
import torch
import torch.nn as nn
from AttackMethod.PackDataset import packDataset_util
from AttackMethod.ModelBased.model_transform import ModelTransform
import pandas as pd
import numpy as np
from random import randint, sample
import random

def load_model(victim_model,data):
    tokenizer = AutoTokenizer.from_pretrained(victim_model)
    if data in ['jigsaw','sst2','agnews']:
        evaluated_model = AutoModelForSequenceClassification.from_pretrained("-".join([victim_model,data]))
    else:
        evaluated_model = AutoModelForSequenceClassification.from_pretrained(victim_model)
    return tokenizer,evaluated_model

def read_sst2(base_path):
    def read_data(file_path):
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        processed_data = []
        for item in data:
            processed_data.append((item[0].strip(), item[1]))
        return processed_data

    train_path = os.path.join(base_path, 'train.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train, test = read_data(train_path), read_data(test_path)
    return train,test

def read_agnews(base_path):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            processed_data.append((item[1].strip() + " " + item[2].strip(), item[0] - 1))
        return processed_data

    train_path = os.path.join(base_path, 'train.csv')
    # dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path), read_data(test_path)
    return train, test


def read_jigsaw(base_path):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            processed_data.append((item[0].strip(), item[1]))
        return processed_data

    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path), read_data(test_path)
    return train, test





def attack(sent, victim_model, tokenizer, transformer):
    batch_size = 128
    aug_sents = transformer.transform(sent[0])
    aug_sents_list = [(each, sent[1]) for each in aug_sents]
    pack_util = packDataset_util(tokenizer)
    aug_sents_loader = pack_util.get_loader(aug_sents_list, shuffle=False, batch_size=batch_size)

    flags = []
    total_correct = 0
    victim_model.eval()
    with torch.no_grad():
        for padded_text, attention_masks, labels in aug_sents_loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = victim_model(padded_text, attention_masks).logits
            _, flag = torch.max(output, dim=1)
            correct = (flag == labels).sum().item()
            flags.extend(flag.tolist())
            total_correct += correct

    victim_model.zero_grad()
    return total_correct / len(aug_sents_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='score', choices=['rule', 'score', 'decision', 'gradient'])
    parser.add_argument('--searching', default='greedy', choices=['greedy', 'pso'])
    parser.add_argument('--degree', type=float, default=-1)  # range from [0, 1]
    parser.add_argument('--attacker', type=str, default='typo',
                        choices=['typo', 'glyph', 'phonetic ', 'synonym', 'contextual', 'inflect', 'syntax', 'distraction'])
    parser.add_argument('--aug_num', type=int, default=100)
    parser.add_argument('--data', default='sst2')
    # parser.add_argument('--size', default='base')
    parser.add_argument('--choice', type=str, default='both', choices=['average', 'worst', 'both'])
    parser.add_argument('--dis_type', default='char')
    parser.add_argument('--victim_model', default='roberta-base')

    params = parser.parse_args()
    degree_1 = params.degree
    mode = params.mode
    searching = params.searching
    attacker = params.attacker
    aug_num = params.aug_num
    data = params.data
    victim_model = params.victim_model
    sent_acc_choice = params.choice
    dis_type = params.dis_type

    base_path = os.path.join('./datasets/', data)
    data_dict = {'sst2':read_sst2,
                'jigsaw':read_jigsaw,
                'agnews':read_agnews}
    train_dataset, test_dataset = data_dict[data](base_path)
    print(len(test_dataset))


    random.seed(123)
    test_dataset = sample(test_dataset, 1000)
    test_num = len(test_dataset)
    print(attacker)
    tokenizer,evaluated_model = load_model(victim_model,data)
    evaluated_model = evaluated_model.cuda()

    evaluated_model.eval()

    rec = []
    ave_save = []
    wst_save = []
    if degree_1 == -1:
        degrees = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        degrees = [degree_1]

    ori_test_dataset = deepcopy(test_dataset)
    for degree in degrees:
        print('degree:', degree)
        if mode == 'rule':
            transformer = load_rule_transformer(attacker, degree=degree, aug_num=aug_num, dataset=train_dataset,
                                                dis_type=dis_type)
        else:
            transformer = ModelTransform(evaluated_model, tokenizer, attacker, degree=degree, dataset=train_dataset,
                                         aug_num=aug_num, dis_type=dis_type)

        right = 0
        temp = []
        score = []

        if sent_acc_choice == "average" or "both":
            for sent in tqdm(ori_test_dataset):
                sent_acc = attack(sent, evaluated_model, tokenizer, transformer)
                score.append(sent_acc)

            print("Average ACC", np.mean(score))
            ave_save.append(np.mean(score))
        if sent_acc_choice == "worst" or "both":
            for sent in tqdm(test_dataset):
                sent_acc = attack(sent, evaluated_model, tokenizer, transformer)
                if sent_acc == 1:
                    right += 1
                    temp.append(sent)
            print("Worst ACC:", right / test_num)
            wst_save.append(right / test_num)
            test_dataset = temp

    from pandas.core.frame import DataFrame

    degrees.append('total score')
    c = {"degree": degrees}
    if sent_acc_choice == "average" or sent_acc_choice == "both":
        total_score = 0
        for i in range(1, len(ave_save)):
            total_score = 0.5 * total_score + 0.5 * ave_save[len(ave_save)-i]
        ave_save.append(total_score)
        c["average"] = ave_save
    if sent_acc_choice == "worst" or sent_acc_choice == "both":
        total_score = 0
        for i in range(1, len(wst_save)):
            total_score = 0.5 * total_score + 0.5 * wst_save[len(wst_save)-i]
        wst_save.append(total_score)
        c["worst"] = wst_save

    data = DataFrame(c)
    print(data)
    save_base_path = './output/'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    data.to_csv(save_base_path+ params.data + "_" + victim_model.split('/')[-1] + "_" + attacker + ".csv", index=False)
