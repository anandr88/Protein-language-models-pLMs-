#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pickle  # Import the pickle module

# Read data from CSV files
train_df = pd.read_csv('train_seq1.csv')  # Update with the correct file path

# Extract sequences from the dataframe
train_sequences = train_df['sequence'].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Encode peptide sequences
train_encodings = tokenizer(train_sequences, truncation=True, padding=True,max_length=35, return_tensors='pt')

# Load pre-trained BERT model
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Forward pass to extract features (embeddings)
with torch.no_grad():
    model.eval()
    outputs = model(**train_encodings)

# Extract the features (embeddings) from the last hidden state
features = outputs.last_hidden_state.numpy()

# Save features to a .pkl file
with open('Bio_bert_features.pkl', 'wb') as f:
    pickle.dump(features, f)

# 'features' now contains the embeddings for each token in your input sequences and is saved in 'features.pkl'

