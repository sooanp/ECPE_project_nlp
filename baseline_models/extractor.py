import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_json_data(file_path, rtype):
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract text and emotion from each utterance in each conversation
    texts = []
    emotions = []
    causes = []
    
    for conversation in data:
        for utterance in conversation['conversation']:
            texts.append(utterance['text'])
            emotions.append(utterance['emotion'])
            causes.append(utterance['cause'])
    
    df = pd.DataFrame({
        'text': texts,
        'label': emotions
    })
    
    df2 = pd.DataFrame({
        'text': texts,
        'label': causes
    })
    if rtype == 'emo':
      return df
    return df2

# load data for text & emo
def text_emo_for_bert(json_file_path):
    df = load_json_data(json_file_path, 'emo')
    return df

# load data for text & cause
def text_cause_for_bert(json_file_path):
    df = load_json_data(json_file_path, 'cause')
    return df


def prepare_data_for_bert(json_file_path, random_state=42):
    # Load data
    df = load_json_data(json_file_path)
    
    # Print dataset statistics
    print(f"Dataset loaded successfully with {len(df)} utterances")
    print("\nEmotion distribution:")
    print(df['label'].value_counts())
    
    # Convert emotions to numeric format
    le = LabelEncoder()
    df['emotion_numeric'] = le.fit_transform(df['label'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, 
        df['emotion_numeric'].values, 
        random_state=random_state,
        stratify=df['emotion_numeric']  # Maintain emotion distribution in train/test splits
    )
    
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': le
    }

if __name__ == "__main__":
    json_file_path = "./data/ecf/train.json"
    
    # Load and prepare data
    processed_data = prepare_data_for_bert(json_file_path)
    
    
    # Print sample information
    print(f"\nNumber of training samples: {len(processed_data['X_train'])}")
    print(f"Number of testing samples: {len(processed_data['X_test'])}")
    print(f"Emotion classes: {list(processed_data['label_encoder'].classes_)}")

    # Print a few examples
    print("\nSample utterances:")
    for i in range(min(5, len(processed_data['df']))):
        text = processed_data['df']['text'].iloc[i]
        emotion = processed_data['df']['label'].iloc[i]
        print(f"Text: '{text}'")
        print(f"Emotion: {emotion}\n")