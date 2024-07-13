#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
from transformers import AutoTokenizer, EsmForSequenceClassification
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import os

# Function to load data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

train_data = load_data("train_seq.csv")
test_data = load_data("test_seq.csv")

# Function to prepare dataset
def prepare_dataset(data):
    sequences = data['sequence'].tolist()
    labels = data['label'].tolist()
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return TensorDataset(inputs.input_ids, inputs.attention_mask, labels)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)

# Split training data into training and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t33_650M_UR50D", num_labels=2)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function for training one epoch
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

# Function for validating the model
def validate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
epochs = 3
for epoch in range(epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
    val_loss = validate_model(model, val_dataloader, loss_fn, device)
    print(f"Epoch {epoch+1}/{epochs} - Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels, logits_list = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_ids = torch.argmax(logits, dim=1)
            predictions.extend(predicted_class_ids.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            logits_list.extend(logits.cpu().numpy())
    return predictions, true_labels, logits_list

# Evaluate the model
predictions, true_labels, logits_list = evaluate_model(model, test_dataloader, device)

# Calculate probabilities for the positive class
logits_tensor = torch.tensor(logits_list)
if logits_tensor.shape[1] == 2:  # 2D logits for both classes
    probs = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
else:  # 1D logits for positive class
    probs = torch.sigmoid(logits_tensor).numpy()

# Calculate performance metrics
confusion = confusion_matrix(true_labels, predictions)
mcc = matthews_corrcoef(true_labels, predictions)
auc = roc_auc_score(true_labels, probs)

# Calculate sensitivity, specificity, and classification report
tn, fp, fn, tp = confusion.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print metrics
print("Confusion Matrix:")
print(confusion)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
print(f"Area Under the Curve (AUC): {auc:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print("Classification Report:")
print(classification_report(true_labels, predictions))

# Save the model
model_save_path = "esm_binary_classification_model"
model.save_pretrained(model_save_path)

# Calculate and print the size of the model in MB
model_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(model_save_path) for filename in filenames) / (1024 * 1024)
print(f"Model Size: {model_size_mb:.2f} MB")

