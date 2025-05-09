import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle
from b1_model import BERTEmotionClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Example usage
if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model, tokenizer, le = load_emotion_classifier()
    print("Model loaded successfully!")
    
    # Example test sentences
    test_sentences = [
        "I am really excited about the new project!",
        "The news was very disappointing.",
        "Why would they do something so frustrating?",
        "It's just another Monday morning.",
        "That food looks disgusting.",
        # Add your own sentences to test
    ]
    
    # Make predictions
    print("\nPredictions:")
    for sentence in test_sentences:
        emotion = predict_emotion(sentence, model, tokenizer, le)
        print(f"Text: '{sentence}'")
        print(f"Predicted emotion: {emotion}\n")
    
    # Interactive mode
    print("\nEnter 'quit' to exit")
    while True:
        user_input = input("\nEnter a sentence to analyze: ")
        if user_input.lower() == 'quit':
            break
        
        emotion = predict_emotion(user_input, model, tokenizer, le)
        print(f"Predicted emotion: {emotion}")