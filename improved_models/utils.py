import json, random
import pandas as pd
import torch
from itertools import product

# Reads the json and returns a dataframe with columns
def load_conversation_data(json_path: str) -> pd.DataFrame:

    with open(json_path, 'r', encoding='utf-8') as f:
        convs = json.load(f)

    records = []
    for conv in convs:
        for utt in conv['conversation']:
            records.append({
                'conversation_ID': conv['conversation_ID'],
                'utterance_ID':    utt['utterance_ID'],
                'text':            utt['text'],
                'emotion':         utt['emotion'],
                'cause':           utt['cause']
            })
    df = pd.DataFrame(records)  
    return df

# Build test pairs for evaluation
def build_test_pairs(convs, model_emo, model_cau, tokenizer,
                     max_len, device, neutral_idx):
    X, y = [], []

    model_emo.eval()
    model_cau.eval()
    with torch.no_grad():
        for conv in convs:
            # Map id→text, gold set
            mp = {u['utterance_ID']: u['text'] for u in conv['conversation']}
            gold = set()
            for emo_ref, cau_ref in conv.get('emotion-cause_pairs', []):
                e_id = int(emo_ref.split('_',1)[0])
                c_id = int(cau_ref.split('_',1)[0])
                gold.add((mp[e_id], mp[c_id]))

            # Emotion, Cause candidates 
            texts = [u['text'] for u in conv['conversation']]
            emo_cands, cau_cands = [], []
            for t in texts:
                inp = tokenizer(
                    t,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_len
                ).input_ids.to(device)

                if model_emo(inp).argmax(1).item() != neutral_idx:
                    emo_cands.append(t)

                if model_cau(inp).argmax(1).item() == 1:
                    cau_cands.append(t)

            # Cartesian pair within same conversation
            for e, c in product(emo_cands, cau_cands):
                if e == c: 
                    continue
                X.append(f"{e} <SEP> {c}")
                y.append(1 if (e, c) in gold else 0)

    return X, y

def build_pair_examples(json_path, neg_ratio=1):
    """
    Returns texts: ['emo_text <SEP> cause_text', ...]
            labels: [1 (valid), 0 (invalid), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X, y = [], []
    for conv in data:
        # Map utterance_ID → text 
        convo_map = {utt['utterance_ID']: utt['text'] for utt in conv['conversation']}

        # Positive examples
        positives = conv.get('emotion-cause_pairs', [])
        for emo_ref, cau_ref in positives:

            emo_id, _     = emo_ref.split('_', 1)
            cau_id, _rest = cau_ref.split('_', 1)

            emo_text   = convo_map.get(int(emo_id))
            cause_text = convo_map.get(int(cau_id)) or _rest

            if emo_text and cause_text:
                X.append(f"{emo_text} <SEP> {cause_text}")
                y.append(1)

        # Negative examples 
        ut_ids = list(convo_map.keys())
        if len(ut_ids) < 2 or not positives:
            continue

        for _ in range(len(positives) * neg_ratio):
            id1, id2 = random.sample(ut_ids, 2)
            X.append(f"{convo_map[id1]} <SEP> {convo_map[id2]}")
            y.append(0)

    return X, y