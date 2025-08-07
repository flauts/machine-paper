#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import dump, load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np
#%% md
# # Load dataset
#%%
from datasets import load_dataset

# Load a custom CSV file
data_files = {"train": "../../data/train_data.csv", "test": "../../data/test_data.csv"}
dataset = load_dataset("csv", data_files=data_files)

# Inspect the first few samples
print(dataset["train"][0])
# Example: Mapping string labels to integers
label_mapping = {
    'notcb': 0,
    'gender': 1,
    'ethnicity': 2,
    'religion': 3,
    'age' : 4,
    'other': 5
}
dataset = dataset.map(lambda x: {"label": label_mapping[x["label"]]})

# Verify the mapping
print(dataset)
#%%
model_name = "distilroberta-base"
#%% md
# # tokenize
#%%
dataset["test"]
#%%
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples): #padding false becase of data collator later
    for text in examples["text"]:
        if text is not None and text != "":
            continue
        else:
            print(text)
            return
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=200)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
print(tokenized_datasets["train"][0])
#%%
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

print(model.config)
#%% md
# # How to Custom Model
#%%
# import torch.nn as nn
#
# class CustomBERTModel(nn.Module):
#     def __init__(self, pretrained_model_name, num_labels):
#         super(CustomBERTModel, self).__init__()
#         self.bert = AutoModel.from_pretrained(pretrained_model_name)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
#
#     def forward(self, input_ids, attention_mask):
#         output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = self.dropout(output[1])  # Applying dropout
#         logits = self.fc(pooled_output)  # Adding a fully connected layer
#         return logits
#
# # Initialize the custom model
# custom_model = CustomBERTModel("bert-base-uncased", num_labels=4)
#%% md
# # Train arguments
#%%
from transformers import TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory for saving model checkpoints
    eval_strategy="epoch",     # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    num_train_epochs=7,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    fp16=True                        # Enable mixed precision for faster training
)

print(training_args)
#%%
from transformers import Trainer
from evaluate import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load a metric (recall and precision for our dataset ania)
metric = load("recall")

# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, f1 with macro averaging (treats all classes equally)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#%%
trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics,    # Custom metric
)
from transformers import EarlyStoppingCallback

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

#%%
trainer.train()
#%%
results = trainer.evaluate()
print(results)
#%%
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
predictions = trainer.predict(tokenized_datasets["test"])
predicted_labels = predictions.predictions.argmax(axis=-1)

# Classification report
print(classification_report(tokenized_datasets["test"]["label"], predicted_labels))

# Confusion matrix
cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nocb", "gender","ethnicity", "religion", "age","other"])
disp.plot(cmap="Blues")  # Optional: set a color map
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)  # You can change the name or dpi as needed
plt.close()  # Close the plot to free memory if you're in a loop
#%%
# Inspect misclassified samples
for idx, (pred, label) in enumerate(zip(predicted_labels, tokenized_datasets["test"]["label"])):
    if pred != label:
        print(f"Index: {idx}, Predicted: {pred}, Actual: {label}")
        print(tokenized_datasets["test"][idx]["text"])
#%%
import os

# Add this after trainer.train() and evaluation
# Save the trained model and tokenizer
os.makedirs("./bert_model", exist_ok=True)
trainer.save_model("../../models/distil_roberta")
tokenizer.save_pretrain("../../models/distil_roberta")
print("Model saved successfully!")