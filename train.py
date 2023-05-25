import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


batch_size = 32
num_epochs = 2
learning_rate = 2e-5
epsilon = 1e-8
max_grad_norm = 1.0
warmup_steps = 0.1


dataset = load_dataset('glue', 'sst2')

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 数据处理函数
def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)


train_dataset = dataset['train'].map(preprocess_function, batched=True)
dev_dataset = dataset['validation'].map(preprocess_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)


optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

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


model.save_pretrained("roberta-base-sst2")


model = RobertaForSequenceClassification.from_pretrained("roberta-base-sst2")
model.to(device)

dev_accuracy = 0
dev_total = 0

with torch.no_grad():
    for batch in dev_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        _, predicted_labels = torch.max(logits, dim=1)
        dev_accuracy += (predicted_labels == labels).sum().item()
        dev_total += labels.size(0)

dev_accuracy /= dev_total
print(f"Dev Accuracy: {dev_accuracy}")
