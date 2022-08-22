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


def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = AutoModelForSequenceClassification.from_pretrained('./models/mbert/finetuned')
    df = pd.read_csv('./data/all-processed/B_english_processed.csv')
    for idx, row in df.iterrows():
        row = df.iloc[idx]
        logits = model(**tokenizer(row.text, return_tensors='pt')).logits
        print(f"{logits.argmax().item()}\t{row.hs}")

    pass


if __name__ == '__main__':
    main()
