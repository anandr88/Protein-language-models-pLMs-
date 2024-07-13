#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix

# Load and preprocess the training data
train_data = pd.read_csv('train_seq1.csv')
train_texts = train_data['sequence'].tolist()
train_labels = train_data['label'].tolist()

# Load and preprocess the testing data
test_data = pd.read_csv('test_seq.csv')
test_texts = test_data['sequence'].tolist()
test_labels = test_data['label'].tolist()

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)

# Tokenize and encode the sequences
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

start_time = time.time() 
# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
end_time = time.time()  # Record end time for testing on training data
testing_time_train = end_time - start_time  # Calculate testing time on training data


start_time = time.time()  # Record start time for testing on training data
# Evaluation
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': None}
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]  # Softmax probabilities for the positive class
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(batch[2].cpu().numpy())

# Calculate metrics
conf_matrix = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(all_labels, all_preds)
#auc = roc_auc_score(all_labels, all_preds)
auc    = roc_auc_score(all_labels, all_probs)
mcc = matthews_corrcoef(all_labels, all_preds)

# Print metrics
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'AUC: {auc:.4f}')
print(f'MCC: {mcc:.4f}')

end_time = time.time()  # Record end time for testing on training data
testing_time_test = end_time - start_time  # Calculate testing time on training data

