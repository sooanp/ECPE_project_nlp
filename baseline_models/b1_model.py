import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import pickle
import extractor
from customDataClass import EmotionDataset, CauseDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "./data/ecf/train_with_cause.json"
test_dir = "./data/ecf/test_with_cause.json"
print(f"Using device: {device}")

# emo data
emo_train_data = extractor.text_emo_for_bert(train_dir)
emo_test_data = extractor.text_emo_for_bert(test_dir)

# cause data
cause_train_data = extractor.text_cause_for_bert(train_dir)
cause_test_data = extractor.text_cause_for_bert(test_dir)

le = LabelEncoder()
emo_train_data['label_numeric'] = le.fit_transform(emo_train_data['label'])
emo_test_data['label_numeric'] = le.fit_transform(emo_test_data['label'])

X_train_emo = np.array(emo_train_data['text'])
X_test_emo = np.array(emo_test_data['text'])
y_train_emo= np.array(emo_train_data['label_numeric'])
y_test_emo = np.array(emo_test_data['label_numeric'])

X_train_cause = np.array(cause_train_data['text'])
X_test_cause = np.array(cause_test_data['text'])
y_train_cause = np.array(cause_train_data['label'])
y_test_causeo = np.array(cause_test_data['label'])


# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets and dataloaders
emo_train_dataset = EmotionDataset(X_train_emo, y_train_emo, tokenizer)
emo_test_dataset = EmotionDataset(X_test_emo, y_test_emo, tokenizer)

cause_train_dataset = CauseDataset(X_train_cause, y_train_cause, tokenizer)
cause_test_dataset = CauseDataset(X_test_cause, y_test_causeo, tokenizer)

batch_size = 8
emo_train_dataloader = DataLoader(emo_train_dataset, batch_size=batch_size, shuffle=True)
emo_test_dataloader = DataLoader(emo_test_dataset, batch_size=batch_size)
cause_train_dataloader = DataLoader(cause_train_dataset, batch_size=batch_size, shuffle=True)
cause_test_dataloader = DataLoader(cause_test_dataset, batch_size=batch_size)

# Emotion Classification model using BERT
class BERTEmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    
# Cause Classification model using BERT
class BERTCauseClassifier(nn.Module):
    def __init__(self):
        super(BERTCauseClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        # Label will be 0 or 1, so output 2
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# Initialize model
num_classes = len(le.classes_)
emo_model = BERTEmotionClassifier(num_classes)
emo_model = emo_model.to(device)

cause_model = BERTCauseClassifier()
cause_model = cause_model.to(device)

# Training parameters
optimizer = AdamW(emo_model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 1  # Reduced for demonstration

# Training function
def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels

# Train the model
# print("Training the emotion model...")
# for epoch in range(epochs):
#     train_loss = train_model(emo_model, emo_train_dataloader, optimizer, loss_fn, device)
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

print("Training the cause model...")
for epoch in range(epochs):
    train_loss = train_model(cause_model, cause_train_dataloader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate the model
# print("\nEvaluating the emotion model...")
# emo_predictions, emo_actual_labels = evaluate_model(emo_model, emo_test_dataloader, device)

print("\nEvaluating the cause model...")
cause_predictions, cause_actual_labels = evaluate_model(cause_model, cause_test_dataloader, device)

# # Print metrics
# print("\nEmotion Classification Report:")
# print(classification_report(emo_actual_labels, emo_predictions, target_names=le.classes_))

print("\nCause Classification Report:")
print(classification_report(cause_actual_labels, cause_predictions))



# # Save the model, tokenizer, and label encoder in a more compatible way
# print("\nSaving the model...")
# save_directory = './saved_models/emotion_classifier_model'

# # Create directory if it doesn't exist
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# # Save model state dict separately to avoid pickling issues
# torch.save(model.state_dict(), f'{save_directory}/model_state_dict.pt')

# # Save other metadata separately using pickle
# metadata = {
#     'num_classes': num_classes,
#     'label_encoder_classes': le.classes_
# }
# with open(f'{save_directory}/metadata.pkl', 'wb') as f:
#     pickle.dump(metadata, f)

# # Save the tokenizer
# tokenizer.save_pretrained(save_directory)

# # Save the label encoder
# with open(f'{save_directory}/label_encoder.pkl', 'wb') as f:
#     pickle.dump(le, f)

# print(f"Model, tokenizer, and label encoder saved to '{save_directory}' directory")