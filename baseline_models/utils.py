import json, random
import pandas as pd

def load_conversation_data(json_path: str) -> pd.DataFrame:
    """
    Reads the JSON and returns a DataFrame with columns:
        'text', 'emotion', 'cause' (0/1)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for conv in data:
        for utt in conv['conversation']:
            records.append({
                'text': utt['text'],
                'emotion': utt['emotion'],
                'cause': utt['cause']
            })
    return pd.DataFrame(records)

def build_pair_examples(json_path, neg_ratio=1):
    """
    Returns texts: ['emo_text <SEP> cause_text', ...]
            labels: [1 (valid), 0 (invalid), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X, y = [], []
    for conv in data:
        # utterance_ID → text mapping
        convo_map = {utt['utterance_ID']: utt['text'] for utt in conv['conversation']}

        # 1) Positive examples
        positives = conv.get('emotion-cause_pairs', [])
        for emo_ref, cau_ref in positives:
            # emo_ref: "4_surprise"  → emo_id="4", _="surprise"
            emo_id, _     = emo_ref.split('_', 1)
            cau_id, _rest = cau_ref.split('_', 1)

            emo_text   = convo_map.get(int(emo_id))
            cause_text = convo_map.get(int(cau_id)) or _rest

            if emo_text and cause_text:
                X.append(f"{emo_text} <SEP> {cause_text}")
                y.append(1)

        # 2) Negative examples (positive당 neg_ratio 개)
        ut_ids = list(convo_map.keys())
        if len(ut_ids) < 2 or not positives:
            continue
        
        for _ in range(len(positives) * neg_ratio):
            id1, id2 = random.sample(ut_ids, 2)
            X.append(f"{convo_map[id1]} <SEP> {convo_map[id2]}")
            y.append(0)

    return X, y