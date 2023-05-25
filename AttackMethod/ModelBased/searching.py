from encodings import search_function
import torch
import torch.nn as nn
import numpy as np
# from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from AttackMethod.PackDataset import packDataset_util

import torch
import numpy as np
from text_visualization import plot_text
# from dataset_IMDB import IMDBPretrainDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, ElectraTokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def integrated_gradients(model, text_token, token_mask, y):
    # 除embedding层外，固定住所有的模型参数
    for name, weight in model.named_parameters():
        if 'embedding' not in name:
            weight.requires_grad = False

    # 获取原始的embedding权重
    # init_embed_weight = model.roberta.word_attn.embedding.weight.data
    # init_embed_weight = model.embeddings.word_embeddings.weight.data
    init_embed_weight = model.roberta.embeddings.word_embeddings.weight.data


    x = text_token

    # 获取输入之后的embedding
    init_word_embedding = init_embed_weight[x[0]]
    # print(init_word_embedding.size())

    # 获取baseline
    baseline = 0 * init_embed_weight
    baseline_word_embedding = baseline[x[0]]

    # 计算线性路径积分
    steps = 50
    # 对目标权重进行线性缩放计算的路径
    gradient_list = []

    for i in range(steps + 1):
        # 进行线性缩放
        scale_weight = baseline + float(i / steps) * (init_embed_weight - baseline)

        # 更换模型embedding的权重
        # model.embeddings.word_embeddings.weight.data = scale_weight
        model.roberta.embeddings.word_embeddings.weight.data = scale_weight
        # init_embed_weight = model.roberta.embeddings.word_embeddings.weight.data


        # 前馈计算
        pred = model(x, token_mask).logits

        # 直接取对应维度的输出(没经过softmax)
        target_pred = pred[:, y]
        # print(target_pred)

        # 计算梯度
        target_pred.backward()

        # 获取输入变量的梯度
        # gradient_list.append(model.embeddings.word_embeddings.weight.grad[x[0]].numpy())
        gradient_list.append(model.roberta.embeddings.word_embeddings.weight.grad[x[0]].numpy())
        # gradient_list.append(model.roberta.embeddings.word_embeddings.weight.grad[x[0]].detach().cpu().numpy())

        # print(gradient_list[-1])
        # 梯度清零，防止累加
        model.zero_grad()

    # steps,input_len,dim
    gradient_list = np.asarray(gradient_list)
    # input_len,dim
    avg_gradient = np.average(gradient_list, axis=0)

    # x-baseline
    delta_x = init_word_embedding - baseline_word_embedding
    delta_x = delta_x.numpy()
    # print(delta_x.shape)

    # 获取积分梯度
    ig = avg_gradient * delta_x

    # 对每一行进行相加得到(input_len,)
    word_ig = np.sum(ig, axis=1)
    return word_ig

    

    
class SearchingMethod():
    def __init__(self, victim, tokenizer,access_info='score', searching_method='greedy'):
        self.access_info = access_info
        self.search_method = searching_method
        self.victim = victim
#         self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer = tokenizer


    def search(self, sample):

        saliency_score = self.access_saliency_score(sample)
        if self.search_method == 'greedy':
            return [k for k, _ in sorted(saliency_score.items(), key=lambda item: item[1])]
        elif self.search_method == 'pso':
            pass
        else:
            print("Invalid searching method")




    def access_saliency_score(self, sample):
        # sentence_tokens = self.tokenizer.tokenize(sample)
        sentence_tokens = sample.split()
        
        return self.sentence_score(sentence_tokens,self.access_info)

        
    def sentence_score(self, sentence_tokens,access_info):
        # target = self.get_pred([self.tokenizer.convert_tokens_to_string(sentence_tokens)])[0]
        target = self.get_pred([' '.join(sentence_tokens)])[0]
        word_losses = {}
        sentence_without = ['']*len(sentence_tokens)
        for i in range(len(sentence_tokens)):
            sentence_tokens_without =  sentence_tokens[:i] +["[MASK]"]+ sentence_tokens[i + 1:]
            sentence_without[i] = ' '.join(sentence_tokens_without)
        if access_info == 'score':
            tempoutput = self.get_prob(sentence_without)
        elif access_info == 'decision':
            tempoutput = self.get_decision(sentence_without)
        elif access_info == 'gradient':
            tempoutput = self.get_gradient(sentence_without,target)
        for i in range(len(sentence_tokens)):
            word_losses[i] = tempoutput[i][target] 
        return word_losses

    def get_pred(self, sentences):
            return self.get_prob(sentences).argmax(axis=1)


    def get_prob(self, sentences):
        batch_size = 200
        # print(len(aug_sents))
        sents_list = [(each,0) for each in sentences]
        
        pack_util = packDataset_util(self.tokenizer)
        sents_loader = pack_util.get_loader(sents_list, shuffle=False, batch_size=batch_size)
        
        outputs =  []
        self.victim.eval()
        with torch.no_grad():
            for padded_text, attention_masks, labels in sents_loader:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels =  padded_text.cuda(),  attention_masks.cuda(),  labels.cuda()
                # print(victim_model.device)
                # print(padded_text.device,attention_masks.device, labels.device)
                output = self.victim(padded_text, attention_masks).logits
                output = output.detach().cpu().tolist()
                # print(output)
                outputs.extend(output)
        # print(outputs[0])
        self.victim.zero_grad()
        return np.array(outputs)
    
    def get_gradient(self,sentence,target):
        model = self.victim
        model.eval()
        tokenizer = self.tokenizer
        labels = []
        text = sentence
        label = target
        encoded_input = tokenizer(text,
                                padding='max_length',  # True:以batch的最长当做最长; max_length: 以指定的当做最长
                                #   max_length=train_data.MAX_LEN,
                                truncation=True,  # padding='max_length',注释掉 truncation
                                )
        input_ids, attention_mask, labels = [], [], []
        input_ids.append(encoded_input.input_ids)
        attention_mask.append(encoded_input.attention_mask)
        labels.append(label)

        # 转Tensor
        input_ids = torch.tensor([i for i in input_ids], dtype=torch.long)
        attention_mask = torch.tensor([a for a in attention_mask], dtype=torch.long)

        # labels = torch.tensor([int(train_data.label_dict[l]) for l in labels], dtype=torch.long)
        labels = torch.tensor([int(l) for l in labels], dtype=torch.long)

        word_ig = integrated_gradients(model, input_ids, attention_mask, labels)

        # 英文由于存在subword的情况，因此会比较麻烦
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        last_index = tokens.index('<pad>')
        tokens = tokens[:last_index]

        word_ig = word_ig[:last_index]

        word_igs = []

        tk = ['']
        igs = [0]
        j = 0
        for i in range(len(tokens)):
            if tokens[i] == '<s>' or tokens[i] == '</s>':
                continue
            if tokens[i].startswith('Ġ'):
                tk.append(tokens[i][1:])
                igs.append(word_ig[i])
                j+=1

            else:
                tk[j]+=tokens[i]
                igs[j]+=word_ig[i]
        igs = [-i for i in igs]
        return igs



    def get_decision(self, sentences):
        pred = self.get_pred(sentences)
        decisions = [0] * len(self.get_prob([sentences])[0])
        decisions[pred] = 1 
        return decisions






