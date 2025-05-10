import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
from b1_model import BERTEmotionClassifier, BERTCauseClassifier
import itertools
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import extractor
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract feature embedding from emo classifier
def get_emo_features(sentences, model, tokenizer):
    model.eval()

    encoding = tokenizer.batch_encode_plus(
        sentences,
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_features=True)

    return outputs

# Extract feature embedding from cause classifier
def get_cause_features(sentences, model, tokenizer):
    model.eval()

    encoding = tokenizer.batch_encode_plus(
        sentences,
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_features=True)

    return outputs


# Load each classifiers
def load_classifiers(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load emotion classifier
    tokenizer = BertTokenizer.from_pretrained(f'{model_dir}/BERT_emo_model/')

    with open(f'{model_dir}/BERT_emo_model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    with open(f'{model_dir}/BERT_emo_model/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    num_classes = metadata['num_classes']

    emo_model = BERTEmotionClassifier(num_classes)
    emo_model.load_state_dict(torch.load(f'{model_dir}/BERT_emo_model/model_state_dict.pt', map_location=device))
    emo_model.to(device)
    emo_model.eval()

    # Load cause classifier
    tokenizer = BertTokenizer.from_pretrained(f'{model_dir}/BERT_cause_model/')

    cause_model = BERTCauseClassifier()
    cause_model.load_state_dict(torch.load(f'{model_dir}/BERT_cause_model/model_state_dict.pt', map_location=device))
    cause_model.to(device)
    cause_model.eval()

    return emo_model, tokenizer, le, cause_model



# Pair classifier model: This is simply a one-fully connected layer
class Pairer(nn.Module):
    def __init__(self, input_dim=1536):
        super(Pairer, self).__init__()
        hidden_dim = 512
        self.input = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        return self.fc(x)


# Cartesian product on features
def prepare_input_and_labels(emo_feats, cause_feats, label_pairs):
    n = len(emo_feats)
    
    pairs = []
    labels = []
    
    for i, j in itertools.product(range(n), range(n)):
        pair_vec = torch.cat([emo_feats[i], cause_feats[j]], dim=0)
        pairs.append(pair_vec)
        
        if [i, j] in label_pairs:
            labels.append(1)
        else:
            labels.append(0)
    

    x = torch.stack(pairs)
    y = torch.tensor(labels).float().unsqueeze(1)
    
    return x, y


def custom_loss(logits, labels, neg_loss_weight):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    losses = bce(logits, labels)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_loss = losses[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=logits.device)
    neg_loss = losses[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=logits.device)

    return pos_loss + neg_loss_weight * neg_loss


if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    emo_model, tokenizer, le, cause_model = load_classifiers("./saved_models")
    print("Model loaded successfully!")
    

    print("Loading Conversation data...")
    conv_train_data , label_train_data = extractor.conv_pair_for_bert("./data/ecf/train_with_cause.json")
    conv_test_data, label_test_data = extractor.conv_pair_for_bert("./data/ecf/test_with_cause.json")
    print("Loading data complete!")


    # print("\nExtracting feature embeddings from Emotion Classifier & Cause Classifier for train data...")
    # train_emotion_features = []
    # train_cause_features = []
    # for conv in tqdm(conv_train_data):
    #     emotion = get_emo_features(conv, emo_model, tokenizer)
    #     cause = get_cause_features(conv, cause_model, tokenizer)
    #     train_emotion_features.append(emotion)
    #     train_cause_features.append(cause)
    # train_label_pairs = label_train_data

    # print("\nExtracting feature embeddings for test data...")
    # test_emotion_features = []
    # test_cause_features = []
    # for conv in tqdm(conv_test_data):
    #     emotion = get_emo_features(conv, emo_model, tokenizer)
    #     cause = get_cause_features(conv, cause_model, tokenizer)
    #     test_emotion_features.append(emotion)
    #     test_cause_features.append(cause)

    # test_label_pairs = label_test_data

    # print("Feature Extraction Complete!")


    # Saved the features to a seperate file
    # with open("cached_train_features.pkl", "wb") as f:
    #     pickle.dump({
    #         "train_emotion_features": train_emotion_features,
    #         "train_cause_features": train_cause_features,
    #         "train_label_pairs": train_label_pairs
    #     }, f)

    # with open("cached_test_features.pkl", "wb") as f:
    #     pickle.dump({
    #         "test_emotion_features": test_emotion_features,
    #         "test_cause_features": test_cause_features,
    #         "test_label_pairs": test_label_pairs
    #     }, f)

    # When Loading the features..
    with open("cached_train_features.pkl", "rb") as f:
        cached_train = pickle.load(f)
        train_emotion_features = cached_train["train_emotion_features"]
        train_cause_features = cached_train["train_cause_features"]
        train_label_pairs = cached_train["train_label_pairs"]

    with open("cached_test_features.pkl", "rb") as f:
        cached_test = pickle.load(f)
        test_emotion_features = cached_test["test_emotion_features"]
        test_cause_features = cached_test["test_cause_features"]
        test_label_pairs = cached_test["test_label_pairs"]

    # Initialize the pair classifier and optimizer
    final_model = Pairer().to(device)
    # criterion = nn.BCEWithLogitsLoss()
    loss_weight = 1.5
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

    final_model.train()
    num_epochs = 5

    print("Training final linear layer for Pair Extraction...")
    for epoch in range(num_epochs):
        total_loss = 0
        for emo_feats, cause_feats, label_pairs in tqdm(zip(train_emotion_features, train_cause_features, train_label_pairs), total=len(train_emotion_features)):
            x, y = prepare_input_and_labels(emo_feats, cause_feats, label_pairs)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = final_model(x)
            # loss = criterion(outputs, y)
            loss = custom_loss(outputs, y, neg_loss_weight=loss_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_emotion_features)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    test_results1 = []
    test_results2 = []
    test_results3 = []
    test_results4 = []
    test_results5 = []

    final_model.eval()

    print("Evaluating...")
    for emo_feats, cause_feats, label_pairs in tqdm(zip(test_emotion_features, test_cause_features, test_label_pairs)):
        x, y = prepare_input_and_labels(emo_feats, cause_feats, label_pairs)
        x, y = x.to(device), y.to(device)
        print("x: ", x.shape)
        print("y: ", y.shape)
        print("y is: ", y)
        with torch.no_grad():
            logits = final_model(x)
            probs = torch.sigmoid(logits)
            
            preds1 = (probs > 0.4).float()
            preds2 = (probs > 0.2).float()
            preds3 = (probs > 0.1).float()
            preds4 = (probs > 0.5).float()
            preds5 = (probs > 0.6).float()
        print("pred: ", preds1.shape)
    
        test_results1.append((preds1.cpu(), y.cpu()))
        test_results2.append((preds1.cpu(), y.cpu()))
        test_results3.append((preds1.cpu(), y.cpu()))
        test_results4.append((preds1.cpu(), y.cpu()))
        test_results5.append((preds1.cpu(), y.cpu()))
    

    
    y_true = torch.cat([y for _, y in test_results1]).numpy()
    y_pred = torch.cat([p for p, _ in test_results1]).numpy()


    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    #######
    y_true = torch.cat([y for _, y in test_results2]).numpy()
    y_pred = torch.cat([p for p, _ in test_results2]).numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    #######
    y_true = torch.cat([y for _, y in test_results3]).numpy()
    y_pred = torch.cat([p for p, _ in test_results3]).numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    #######
    y_true = torch.cat([y for _, y in test_results4]).numpy()
    y_pred = torch.cat([p for p, _ in test_results4]).numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    #######
    y_true = torch.cat([y for _, y in test_results5]).numpy()
    y_pred = torch.cat([p for p, _ in test_results5]).numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")






    # train_X, train_y = prepare_input_and_labels(train_emotion_features, train_cause_features, train_label_pairs)
    # test_X, test_y  = prepare_input_and_labels(test_emotion_features, test_cause_features, test_label_pairs)
    # print(test_X.shape)
    # print(test_y.shape)
    # final_model = Pairer()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

    # final_model.train()
    # print("Training final liner layer for Pair Extraction...")
    # for epoch in range(3):
    #     optimizer.zero_grad()
    #     outputs = final_model(train_X)
    #     loss = criterion(outputs, train_y)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    

    # # Evaluation
    # final_model.eval()
    # with torch.no_grad():
    #     outputs = final_model(test_X)
    #     probs = torch.sigmoid(outputs)
    #     preds = (probs > 0.5).float()

    # y_true = test_y.cpu().numpy()
    # y_pred = preds.cpu().numpy()

    # acc = accuracy_score(y_true, y_pred)
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")