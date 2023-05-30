import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os

import argparse

import torch
from torch.utils.data import Dataset


batch_size = 32
num_epochs = 5
learning_rate = 2e-5
epsilon = 1e-8
max_grad_norm = 1.0
warmup_steps = 0.1



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids, mask,label = self.data[index]
        # print(ids.type)
        return torch.tensor(ids), torch.tensor(mask),label


def token_text(text,tokenizer):
    return tokenizer(text, padding='max_length', truncation=True)

def read_sst2(base_path,tokenizer):
    def read_data(file_path):
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        processed_data = []
        for item in data:
            tk = token_text(item[0].strip(),tokenizer)
            processed_data.append((tk['input_ids'],tk['attention_mask'], item[1]))
        return processed_data

    train_path = os.path.join(base_path, 'train.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train, test = read_data(train_path), read_data(test_path)
    return train,test


def read_agnews(base_path,tokenizer):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            tk = token_text(item[1].strip() + " " + item[2].strip(),tokenizer)
            processed_data.append((tk['input_ids'],tk['attention_mask'], item[0] - 1))
        return processed_data

    train_path = os.path.join(base_path, 'train.csv')
    # dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path), read_data(test_path)
    return train, test


def read_jigsaw(base_path,tokenizer):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            tk = token_text(item[0].strip(),tokenizer)
            processed_data.append((tk['input_ids'],tk['attention_mask'], item[1]))
        return processed_data

    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path), read_data(test_path)
    return train, test

def train_and_eval(data_name,model_type):
    data_name = data_name
    model_name = 'roberta-'+model_type
    print(data_name)
    print(model_name)
    if data_name == 'agnews':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_path = os.path.join('./datasets/', data_name)
    data_dict = {'sst2':read_sst2,
                'jigsaw':read_jigsaw,
                'agnews':read_agnews}
    train_data, test_data = data_dict[data_name](base_path,tokenizer)


    def collate_fn(batch):
        ids, mask, labels = zip(*batch)
        ids = torch.stack(ids, dim=0)
        mask = torch.stack(mask, dim=0)
        labels = torch.tensor(labels)
        return ids,mask, labels


    # 构建dataloader
    train_dataloader = torch.utils.data.DataLoader(MyDataset(train_data), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    test_dataloader  = torch.utils.data.DataLoader(MyDataset(test_data), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)




    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_ids,attention_mask,labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # print(input_ids.size())
            # print(attention_mask.size())
            # print(labels.size())
            

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss}")

    model.eval()


    model.save_pretrained("-".join(["roberta",model_type,data_name]))


    model = AutoModelForSequenceClassification.from_pretrained("-".join(["roberta",model_type,data_name]))
    model.to(device)

    dev_accuracy = 0
    dev_total = 0

    with torch.no_grad():
        for input_ids,attention_mask,labels in test_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            _, predicted_labels = torch.max(logits, dim=1)
            dev_accuracy += (predicted_labels == labels).sum().item()
            dev_total += labels.size(0)

    dev_accuracy /= dev_total
    print(f"Dev Accuracy: {dev_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='sst2', choices=['sst2', 'jigsaw','agnews'])
    parser.add_argument('--model_type', default='base', choices=['base', 'large'])

    params = parser.parse_args()

    model_type = params.model_type
    data_name= params.data_name

    train_and_eval(data_name,model_type)