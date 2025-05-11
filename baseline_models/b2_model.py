import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import GPT2TokenizerFast
from collections import Counter
from utils import load_conversation_data, build_pair_examples, build_test_pairs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_train = load_conversation_data("./data/ecf/train_dev_with_cause.json")

df_test = load_conversation_data("./data/ecf/test_with_cause.json")

with open("./data/ecf/train_dev_with_cause.json", 'r', encoding='utf-8') as f:
    raw = json.load(f)

# Convert labels to numeric format
le = LabelEncoder()
df_train['label_emo'] = le.fit_transform(df_train['emotion'])
df_test['label_emo'] = le.fit_transform(df_test['emotion'])
neutral_idx = list(le.classes_).index("neutral")

# Cause to int
df_train['label_cause'] = df_train['cause'].astype(int)
df_test['label_cause'] = df_test['cause'].astype(int)

# Create train, test set
X_train     = df_train['text'].values
y_emo_train = df_train['label_emo'].values
y_cau_train = df_train['label_cause'].values

X_test      = df_test ['text'].values
y_emo_test  = df_test ['label_emo'].values
y_cau_test  = df_test ['label_cause'].values

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
batch_size = 16

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

        input_ids = encoding['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load Dataset and Dataloader
emo_train_ds = EmotionDataset(X_train, y_emo_train, tokenizer, max_len)
emo_test_ds  = EmotionDataset(X_test,  y_emo_test,  tokenizer, max_len)
cau_train_ds = EmotionDataset(X_train, y_cau_train, tokenizer, max_len)
cau_test_ds  = EmotionDataset(X_test,  y_cau_test,  tokenizer, max_len)

emo_train_loader = DataLoader(emo_train_ds, batch_size=batch_size, shuffle=True)
emo_test_loader  = DataLoader(emo_test_ds,  batch_size=batch_size)
cau_train_loader = DataLoader(cau_train_ds, batch_size=batch_size, shuffle=True)
cau_test_loader  = DataLoader(cau_test_ds,  batch_size=batch_size)

# LSTM Emotion Classifier 
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
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

# Training & Evaluation
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

epochs = 20
for epoch in range(epochs):
    loss_e = train_epoch(model_emo, emo_train_loader, optimizer_emo, loss_fn)
    loss_c = train_epoch(model_cau, cau_train_loader, optimizer_cau, loss_fn)
    print(f"[Epoch {epoch+1}/{epochs}] Emo Loss: {loss_e:.4f} | Cau Loss: {loss_c:.4f}")

emo_preds, emo_trues = eval_model(model_emo, emo_test_loader)
cau_preds, cau_trues = eval_model(model_cau, cau_test_loader)

print("\nEmotion Classification Report")
print(classification_report(emo_trues, emo_preds, target_names=le.classes_))
print("Cause Classification Report")
print(classification_report(cau_trues, cau_preds, target_names=['no_cause','cause']))

# Filtering emotion and cause pair
class Pairer(nn.Module):
    def __init__(self, input_dim=512):   

        super().__init__()
        hidden_dim = 512
        self.input = nn.Linear(input_dim, hidden_dim)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = self.input(x)
        x = self.relu(x)
        return self.fc(x).squeeze(1)

Xp_tr, yp_tr = build_pair_examples("./data/ecf/train_dev_with_cause.json")

Xp_te, yp_te = build_test_pairs(
    raw, model_emo, model_cau,
    tokenizer, max_len, device, neutral_idx
)

# Weights for classs imbalance
cnts = Counter(yp_tr)
total = sum(cnts.values())
weights = torch.tensor([total/cnts[0], total/cnts[1]], device=device)

train_pf = EmotionDataset(Xp_tr, yp_tr, tokenizer, max_len)
test_pf  = EmotionDataset(Xp_te, yp_te, tokenizer, max_len)
pf_tr_ld = DataLoader(train_pf, batch_size, shuffle=True)
pf_te_ld = DataLoader(test_pf,  batch_size)

model_f = Pairer(input_dim=128*2*2).to(device)      
optimizer = AdamW(model_f.parameters(), lr=5e-2)
loss_fn_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights[1]))

# Get last hidden state from LSTMClassifier
def get_last_hidden(model, tokenizer, text, max_len, device):
    enc = tokenizer(text,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                   ).input_ids.to(device)        
    with torch.no_grad():
        
        lengths = (enc != pad_idx).sum(1)
        x = model.embedding(enc)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                   batch_first=True,
                                                   enforce_sorted=False)
        out_p, _ = model.lstm(packed)
        out, _   = nn.utils.rnn.pad_packed_sequence(out_p,
                                                    batch_first=True)
        idx = (lengths-1).view(-1,1,1).expand(-1,1,out.size(2))
        last = out.gather(1, idx).squeeze(1)       
    return last.squeeze(0)                        

# Pair-Feature Dataset
class PairFeatureDataset(Dataset):
    def __init__(self, texts, labels,
                 model_emo, model_cau,
                 tokenizer, max_len, device):
        self.texts     = texts
        self.labels    = labels
        self.model_emo = model_emo
        self.model_cau = model_cau
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.device    = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        e, c = self.texts[idx].split(' <SEP> ')
        h_e  = get_last_hidden(self.model_emo,
                               self.tokenizer,
                               e, self.max_len, self.device)
        h_c  = get_last_hidden(self.model_cau,
                               self.tokenizer,
                               c, self.max_len, self.device)
        feat = torch.cat([h_e, h_c], dim=0)         
        label= torch.tensor(self.labels[idx], dtype=torch.float)
        return feat, label

# Instantiate Pairer and DataLoaders
model_f = Pairer(input_dim=128*2*2).to(device)      
optimizer = AdamW(model_f.parameters(), lr=1e-3)
loss_fn_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights[1]))

train_ds = PairFeatureDataset(Xp_tr, yp_tr,
                              model_emo, model_cau,
                              tokenizer, max_len, device)
test_ds  = PairFeatureDataset(Xp_te, yp_te,
                              model_emo, model_cau,
                              tokenizer, max_len, device)

train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld  = DataLoader(test_ds,  batch_size=batch_size)

# Training & Evaluation
for epoch in range(epochs):
    model_f.train()
    total_loss = 0
    for feats, lbls in train_ld:
        feats = feats.to(device)
        lbls  = lbls.to(device)
        logits= model_f(feats)                    
        loss  = loss_fn_f(logits, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Filter-Epoch {epoch+1}/{epochs}] Loss={total_loss/len(train_ld):.4f}")

model_f.eval()
preds, trues = [], []
with torch.no_grad():
    for feats, lbls in test_ld:
        feats = feats.to(device)
        out   = torch.sigmoid(model_f(feats))   
        pred  = (out>0.5).long().cpu().tolist()
        preds.extend(pred)
        trues.extend(lbls.cpu().long().tolist())

print("==Filter==")
print(classification_report(trues, preds, target_names=['neg','pos']))
print("ConfMat\n", confusion_matrix(trues, preds))
