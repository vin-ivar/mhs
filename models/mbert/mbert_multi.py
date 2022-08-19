import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import datasets
from datasets import load_dataset, Dataset, load_metric
from tqdm.auto import tqdm

MODEL_NAME = 'bert-base-multilingual-cased'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    if example['text'] is None:
        return tokenizer('', truncation=True, padding='max_length')
    return tokenizer(example['text'], truncation=True, padding='max_length')

# %%
def load_and_tokenize_dataset(csv_file: str):
    dataset = load_dataset('csv', data_files=csv_file)
    print(dataset['train'].column_names)
    has_unnamed_col = 'Unnamed: 0' in dataset['train'].column_names
    if has_unnamed_col:
        dataset = dataset.rename_column('Unnamed: 0', 'idx')
    dataset = dataset['train'].train_test_split(test_size=0.2)

    tokenized_datasets = dataset.map(tokenize_function)
    for dataset in ['train', 'test']:
        if 'id' in tokenized_datasets[dataset].column_names:
            tokenized_datasets[dataset] = tokenized_datasets[dataset].remove_columns(['id'])
        if has_unnamed_col:
            tokenized_datasets[dataset] = tokenized_datasets[dataset].remove_columns(['text', 'idx', 'token_type_ids'])
        else:
            tokenized_datasets[dataset] = tokenized_datasets[dataset].remove_columns(['text', 'token_type_ids'])
        tokenized_datasets[dataset] = tokenized_datasets[dataset].rename_column('hs', 'labels')
        tokenized_datasets[dataset].set_format('torch')
    return tokenized_datasets


# %%
languages = ['arabic', 'danish', 'english', 'french', 'german', 'hindi', 'indonesian', 'italian', 'portuguese',
             'spanish', 'turkish']
# %%
ds = []
for lang in languages:
    df = pd.read_csv(
        f'https://raw.githubusercontent.com/vidhur2k/Multilngual-Hate-Speech/main/data/all-processed/B_{lang}_processed.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    ds.append(df)

train_datasets = []
test_datasets = {}
for i in range(len(languages)):
    dataset = Dataset.from_pandas(ds[i])
    dataset = dataset.train_test_split(test_size=0.2)
    train_datasets.append(dataset['train'])
    test_datasets[languages[i]] = dataset['test']

train_dataset = datasets.concatenate_datasets(train_datasets)
tokenized_dataset = train_dataset.map(tokenize_function)
tokenized_dataset = tokenized_dataset.remove_columns(['text', 'token_type_ids'])
tokenized_dataset = tokenized_dataset.rename_column('hs', 'labels')
tokenized_dataset.set_format('torch')

#%%
def get_train_loader(tokenized_dataset, batch_size):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    return train_loader


# Define training hyperparameters for the monolingual scenario
n_epochs = 5
lr = 5e-5
batch_size = 64

def train(train_loader):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    n_training_steps = n_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_training_steps,
    )

    progress = tqdm(range(n_training_steps))
    model.train()

    for epoch in range(n_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress.update(1)

    return model


def test(test_loader, model):
    progress = tqdm(range(len(test_loader)))
    accuracy_metric = load_metric("accuracy")
    model.to(device)
    model.eval()
    preds = []
    trues = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
        preds.extend(predictions.tolist())
        trues.extend(batch['labels'].tolist())
        progress.update(1)

    print(accuracy_metric.compute())
    print(f'F1 Score: {f1_score(trues, preds, average="weighted")}')
    print(f'AUC Score: {roc_auc_score(trues, preds, average="weighted")}')


# %%
train_loader = get_train_loader(tokenized_dataset, batch_size=8)
multilingual_model = train(train_loader)
multilingual_model.save_pretrained('./finetuned.pt')
