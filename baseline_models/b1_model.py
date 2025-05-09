import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = EmotionDataset(X_train, y_train, tokenizer)
test_dataset = EmotionDataset(X_test, y_test, tokenizer)

batch_size = 8  # Small batch size for demonstration purposes
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Define the Emotion Classification model using BERT
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

# Initialize model
num_classes = len(le.classes_)
model = BERTEmotionClassifier(num_classes)
model = model.to(device)

# Training parameters
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 5  # Reduced for demonstration

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
print("Training the model...")
for epoch in range(epochs):
    train_loss = train_model(model, train_dataloader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate the model
print("\nEvaluating the model...")
predictions, actual_labels = evaluate_model(model, test_dataloader, device)

# Print metrics
print("\nClassification Report:")
print(classification_report(actual_labels, predictions, target_names=le.classes_))

# Function to predict emotions for new sentences
def predict_emotion(text, model, tokenizer, le):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    predicted_class = le.inverse_transform([preds.item()])[0]
    return predicted_class

# Function to load the saved model and make predictions
def load_emotion_classifier(model_dir='emotion_classifier_model'):
    """
    Load the saved BERT emotion classifier model and return all necessary components
    for making predictions.
    
    Args:
        model_dir (str): Directory where the model files are saved
        
    Returns:
        model: Loaded BERT emotion classifier model
        tokenizer: BERT tokenizer
        le: Label encoder for emotion classes
    """
    import os
    import pickle
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model state
    checkpoint = torch.load(f'{model_dir}/bert_emotion_classifier.pth', map_location=device)
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    # Load the label encoder
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Initialize the model with the right number of classes
    num_classes = checkpoint['num_classes']
    model = BERTEmotionClassifier(num_classes)
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer, le

# Save the model, tokenizer, and label encoder in a more compatible way
print("\nSaving the model...")
save_directory = './saved_models/emotion_classifier_model'
import os

# Create directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save model state dict separately to avoid pickling issues
torch.save(model.state_dict(), f'{save_directory}/model_state_dict.pt')

# Save other metadata separately using pickle
import pickle
metadata = {
    'num_classes': num_classes,
    'label_encoder_classes': le.classes_
}
with open(f'{save_directory}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

# Save the label encoder
with open(f'{save_directory}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Model, tokenizer, and label encoder saved to '{save_directory}' directory")

# Example usage with new sentences
test_sentences = [
    "I am really excited about the new project!",
    "The news was very disappointing.",
    "Why would they do something so frustrating?",
    "It's just another Monday morning.",
    "That food looks disgusting."
]

print("\nPredictions for new sentences:")
for sentence in test_sentences:
    emotion = predict_emotion(sentence, model, tokenizer, le)
    print(f"Text: '{sentence}'\nPredicted emotion: {emotion}\n")