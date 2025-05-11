# ECPE_project_nlp
CSE538 Course Final Project

# Team members
Sunwoo Kim, Jungmin Park and Sooan Park

# NLP concepts used
1. Syntax | Classification: Classifying emotions, causes and (emotion, cause) pair
2. Semantics | Probabilistic Model: Embedding for emotions, causes and (emotion, cause) pair representation
3. Language Modeling | Transformers: Bert and LSTM
4. Applications | Custom Statistical or Symbolic: Language and Psychology (for emotions)

# General Description of the code
Our code implements the model for predicting emotion-cause pair in dialogue data.

Work flow:
1. load and preprocess data
2. Train emotion and cause classifiers
3. Generate utterance pairs, extract features via two classifiers, and train the filter(Pairer)

# System for running code
We used Google Cloud Platform runnign Ubuntu to run the code.