import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import GPT2TokenizerFast
import pickle
import os

from utils import load_conversation_data, build_pair_examples

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = load_conversation_data("./data/ecf/train_with_cause.json")

# Convert labels to numeric format
le = LabelEncoder()
df['label_emo'] = le.fit_transform(df['emotion'])

# Cause to int
df['label_cause'] = df['cause'].astype(int)

# Split data into train and test sets
X_train, X_test, y_emo_train, y_emo_test, y_cau_train, y_cau_test = train_test_split(
    df['text'].values,
    df['label_emo'].values,
    df['label_cause'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label_emo']
)

# Load BPE (GPT2) tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    'bos_token': '<s>',
    'eos_token': '</s>'
})
tokenizer.pad_token = '<|endoftext|>'

vocab_size = tokenizer.vocab_size
pad_idx    = tokenizer.pad_token_id
max_len = 128

# Custom Dataset class for lstm
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # flatten batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

batch_size = 16
emo_train_ds = EmotionDataset(X_train, y_emo_train, tokenizer, max_len)
emo_test_ds  = EmotionDataset(X_test,  y_emo_test,  tokenizer, max_len)
cau_train_ds = EmotionDataset(X_train, y_cau_train, tokenizer, max_len)
cau_test_ds  = EmotionDataset(X_test,  y_cau_test,  tokenizer, max_len)

emo_train_loader = DataLoader(emo_train_ds, batch_size=batch_size, shuffle=True)
emo_test_loader  = DataLoader(emo_test_ds,  batch_size=batch_size)
cau_train_loader = DataLoader(cau_train_ds, batch_size=batch_size, shuffle=True)
cau_test_loader  = DataLoader(cau_test_ds,  batch_size=batch_size)

# ========== 3. LSTM Emotion Classifier ==========
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, input_ids):
        lengths = (input_ids != pad_idx).sum(dim=1)
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_p, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        idx = (lengths-1).view(-1,1,1).expand(-1,1,out.size(2))
        last_hidden = out.gather(1, idx).squeeze(1)
        logits = self.fc(self.dropout(last_hidden))
        return logits

# Instantiate emotion & cause classifiers
num_emo_classes = len(le.classes_)
model_emo = LSTMClassifier(vocab_size, 128, 128, num_emo_classes, pad_idx).to(device)
model_cau = LSTMClassifier(vocab_size, 128, 128, 2, pad_idx).to(device)

optimizer_emo = AdamW(model_emo.parameters(), lr=1e-3)
optimizer_cau = AdamW(model_cau.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ========== 4. Training & Evaluation ==========
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels    = batch['label'].to(device)
        logits    = model(input_ids)
        loss      = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['label'].to(device)
            logits    = model(input_ids)
            _, batch_preds = torch.max(logits, dim=1)
            preds.extend(batch_preds.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    return preds, trues

epochs = 5
for epoch in range(1, epochs+1):
    loss_e = train_epoch(model_emo, emo_train_loader, optimizer_emo, loss_fn)
    loss_c = train_epoch(model_cau, cau_train_loader, optimizer_cau, loss_fn)
    print(f"[Epoch {epoch}/{epochs}] Emo Loss: {loss_e:.4f} | Cau Loss: {loss_c:.4f}")

emo_preds, emo_trues = eval_model(model_emo, emo_test_loader)
cau_preds, cau_trues = eval_model(model_cau, cau_test_loader)

print("\nEmotion Classification Report")
print(classification_report(emo_trues, emo_preds, target_names=le.classes_))
print("Cause Classification Report")
print(classification_report(cau_trues, cau_preds, target_names=['no_cause','cause']))

# ========== 6. Cartesian Product of Predicted Clauses ==========
def cartesian_pairs(emotions, causes):
    from itertools import product
    return list(product(emotions, causes))

# Inference: collect predicted emotion & cause clauses
model_emo.eval()
model_cau.eval()
emo_clauses, cau_clauses = [], []

with torch.no_grad():
    for text in df['text'].values:
        enc = tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        # emotion pred
        log_emo = model_emo(input_ids)
        if log_emo.argmax(1).item() > 0:
            emo_clauses.append({'text': text})
        # cause pred
        log_cau = model_cau(input_ids)
        if log_cau.argmax(1).item() == 1:
            cau_clauses.append({'text': text})

all_pairs = cartesian_pairs(emo_clauses, cau_clauses)
print(f"Total candidate pairs: {len(all_pairs)}")

X_pf, y_pf = build_pair_examples('./data/ecf/train.json', neg_ratio=1)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_pf, y_pf, test_size=0.2, random_state=42, stratify=y_pf
)

class PairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

pf_train_ds = PairDataset(Xp_train, yp_train, tokenizer, max_len)
pf_test_ds  = PairDataset(Xp_test,  yp_test,  tokenizer, max_len)
pf_train_loader = DataLoader(pf_train_ds, batch_size=batch_size, shuffle=True)
pf_test_loader  = DataLoader(pf_test_ds,  batch_size=batch_size)

# instantiate filter model
model_filter = LSTMClassifier(vocab_size, 128, 128, 2, pad_idx).to(device)
opt_filt = AdamW(model_filter.parameters(), lr=1e-3)

# train filter
for epoch in range(1, epochs+1):
    loss_f = train_epoch(model_filter, pf_train_loader, opt_filt, loss_fn)
    print(f"[Filter Epoch {epoch}/{epochs}] Loss: {loss_f:.4f}")

# evaluate filter
pf_preds, pf_trues = eval_model(model_filter, pf_test_loader)
print("\nFilter Classification Report")
print(classification_report(pf_trues, pf_preds, target_names=['invalid','valid']))

# ========== 8. Final Inference Pipeline ==========
def find_emotion_cause_pairs(texts):
    # predict emotion & cause clauses
    emo_list, cau_list = [], []
    for t in texts:
        enc = tokenizer.encode_plus(
            t, add_special_tokens=True,
            max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        ids = enc['input_ids'].to(device)
        if model_emo(ids).argmax(1).item() > 0:
            emo_list.append(t)
        if model_cau(ids).argmax(1).item() == 1:
            cau_list.append(t)
    # cartesian
    candidates = cartesian_pairs(
        [{'text':e} for e in emo_list],
        [{'text':c} for c in cau_list]
    )
    # filter
    valid = []
    for e,c in candidates:
        pair_text = e['text'] + ' <SEP> ' + c['text']
        enc = tokenizer.encode_plus(
            pair_text, add_special_tokens=True,
            max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        pred = model_filter(enc['input_ids'].to(device)).argmax(1).item()
        if pred == 1:
            valid.append((e['text'], c['text']))
    return valid

# usage
valid_pairs = find_emotion_cause_pairs(list(df['text'].head(10).values))
print("Valid emotion–cause pairs:", valid_pairs)
