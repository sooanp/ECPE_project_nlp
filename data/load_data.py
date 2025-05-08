from datasets import load_dataset

# Dataset for pretraining emotion classification
RECCON = load_dataset("roskoN/dailydialog", cache_dir='./reccon')   