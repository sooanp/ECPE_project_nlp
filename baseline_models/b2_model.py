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

import extractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = extractor.text_emo_for_bert("./data/ecf/train.json")
# Convert labels to numeric format
le = LabelEncoder()
df['label_numeric'] = le.fit_transform(df['label'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values, df['label_numeric'].values, test_size=0.2, random_state=42, stratify=df['label_numeric']
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
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
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
            'label': torch.tensor(label, dtype=torch.long)
        }

batch_size = 16
train_ds = EmotionDataset(X_train, y_train, tokenizer, max_len=128)
test_ds  = EmotionDataset(X_test,  y_test,  tokenizer, max_len=128)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size)

# ========== 3. LSTM Emotion Classifier ==========
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, input_ids):
        lengths = (input_ids != pad_idx).sum(dim=1)
        x = self.embedding(input_ids)  # [B, L, E]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        idx = (lengths - 1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(1)
        last_hidden = out.gather(1, idx).squeeze(1)  # [B, hidden*2]
        logits = self.fc(self.dropout(last_hidden))
        return logits

num_classes = len(le.classes_)
model = LSTMEmotionClassifier(vocab_size, embed_dim=128, hidden_dim=128,
                              num_classes=num_classes, pad_idx=pad_idx).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

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

epochs = 30
for epoch in range(1, epochs+1):
    loss = train_epoch(model, train_loader, optimizer, loss_fn)
    print(f"[Epoch {epoch}/{epochs}] train loss: {loss:.4f}")

preds, trues = eval_model(model, test_loader)
print("\nClassification Report:")
print(classification_report(trues, preds, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(trues, preds))

# ========== 5. Inference ==========
def predict_emotion(text: str, model, tokenizer, le, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    with torch.no_grad():
        logits = model(input_ids)
        _, pred = torch.max(logits, dim=1)
    return le.inverse_transform([pred.item()])[0]

# ========== 6. Save Artifacts ==========
save_dir = "./saved_models/lstm_emotion_gpt2"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), f"{save_dir}/model_state_dict.pt")
tokenizer.save_pretrained(save_dir)
with open(f"{save_dir}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
meta = {"num_classes": num_classes, "pad_idx": pad_idx, "max_len": max_len}
with open(f"{save_dir}/metadata.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"Artifacts saved to {save_dir}")

# ========== 7. Usage Example ==========
for sent in ["I feel ecstatic today!", "This is so frustrating..."]:
    print(f"'{sent}' →", predict_emotion(sent, model, tokenizer, le))