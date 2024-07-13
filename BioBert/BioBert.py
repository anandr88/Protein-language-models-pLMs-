#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import time
import joblib

# Read data from CSV files
train_df = pd.read_csv('train_seq1.csv')  # Update with the correct file path
test_df = pd.read_csv('test_seq.csv')    # Update with the correct file path

# Extract sequences and labels from the dataframes
train_sequences = train_df['sequence'].tolist()
train_labels = train_df['label'].tolist()
test_sequences = test_df['sequence'].tolist()
test_labels = test_df['label'].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Encode peptide sequences
train_encodings = tokenizer(train_sequences, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_sequences, truncation=True, padding=True, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(train_labels).long()
test_labels = torch.tensor(test_labels).long()

# Split data into train and test sets
train_inputs, test_inputs, train_masks, test_masks = train_encodings.input_ids, test_encodings.input_ids, \
                                                      train_encodings.attention_mask, test_encodings.attention_mask

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=2)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Training loop
batch_size = 16
epochs = 3

start_time = time.time()  # Start training time measurement

for epoch in range(epochs):
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i + batch_size]
        batch_masks = train_masks[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

end_time = time.time()  # End training time measurement
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Save the model to a .pkl file
model_path = "biobert_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Print the size of the model
model_size = joblib.os.path.getsize(model_path)
print(f"Model size: {model_size} bytes")

# Evaluation on test set
model.eval()
with torch.no_grad():
    outputs = model(test_inputs, attention_mask=test_masks)
    logits = outputs.logits
    predictions = np.argmax(logits.cpu().numpy(), axis=1)

# Classification report
print(classification_report(test_labels, predictions))

# Calculate additional evaluation metrics
accuracy = accuracy_score(test_labels, predictions)
mcc = matthews_corrcoef(test_labels, predictions)
auc = roc_auc_score(test_labels, logits[:, 1])  # Assuming the second class is the positive class
tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy}")
print(f"MCC: {mcc}")
print(f"AUC: {auc}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")


# In[ ]:




