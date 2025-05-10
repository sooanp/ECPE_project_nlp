import json
import pandas as pd
import re
def load_json_data(file_path, rtype):
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


# Get conversation [u_1, u_2, ... u_n] ans [[e_1, c_1]....] as labels
# u is utterance and e, c are utterance ID for each emotions and causes
def conv_pair_for_bert(json_file_path):
      with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

      conversation_list = []
      label_pairs = []

      for conversation in data:
            conv_utts = [utt['text'] for utt in conversation['conversation']]
            pairs = conversation.get('emotion-cause_pairs', [])

            # Convert string pairs
            pattern = r'\d+'
            processed_pairs = []
            for e,c in pairs:
                e_id = int(re.match(pattern, e).group())
                c_id = int(re.match(pattern, c).group())
                processed_pairs.append([e_id, c_id])
            
            conversation_list.append(conv_utts)
            label_pairs.append(processed_pairs)

      return conversation_list, label_pairs

# def main():
#     c, l = conv_pair_for_bert("./data/ecf/train_with_cause.json")
#     print(c[:1])
#     print(l[:1])

# if __name__== "__main__":
#     main()