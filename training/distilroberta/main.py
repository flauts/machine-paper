#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import dump, load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, set_seed
from datasets import load_dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np
import evaluate
#%%
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
#%% md
# # Load dataset
#%%
from datasets import load_dataset

# Load a custom CSV file
data_files = {"train": "../../data/train_data.csv", "valid":"../../data/val_data.csv","test": "../../data/test_data.csv"}
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
dataset["valid"]
#%%
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples): #padding false becase of data collator later
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=200)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
print(tokenized_datasets["train"][0])
#%%
metric = evaluate.load("recall")


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    print(model.config)
    return model
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
# ## hyperparameter tuning
#%%
import optuna
from optuna.storages import RDBStorage
import os

# Define persistent storage
storage = RDBStorage("sqlite:///optuna_trials.db")

study = optuna.create_study(
    study_name=f"{model_name}-opt-study", direction="maximize", storage=storage, load_if_exists=True
)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="cyberbullying-bert-based-finetuning"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#%%
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments


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

def compute_objective(metrics):
    return metrics["eval_recall"]


wandb.init(project="cyberbullying-bert-based-finetuning", name=f"{model_name}-opt-study")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    logging_strategy="epoch",
    num_train_epochs=3,
    report_to="wandb",
    logging_dir="./logs",
    # run_name=f"{model_name}-opt-study",
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#%%
def optuna_hp_space(trial):
    return {
        "learning_rate" : trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True),
        "per_device_train_batch_size" : trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "num_train_epochs" : trial.suggest_int("num_train_epochs", 5, 10),
        "warmup_steps" : trial.suggest_int("warmup_steps", 0, 500),
        "weight_decay" : trial.suggest_float("weight_decay", 0.01, 0.1)
}


best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
    compute_objective=compute_objective,
    study_name=f"{model_name}-opt-study",
    storage="sqlite:///optuna_trials.db",
    load_if_exists=True,
)

print(best_run)
#%%
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
)
import matplotlib.pyplot as plt

# Load the study from RDB storage
storage = optuna.storages.RDBStorage("sqlite:///optuna_trials.db")

study = optuna.load_study(study_name=f"{model_name}-opt-study", storage=storage)

# Plot optimization history
ax1 = plot_optimization_history(study)
plt.show()
ax1.figure.savefig("optimization_history.png")

# Plot intermediate values (if using pruning and intermediate reports)
ax2 = plot_intermediate_values(study)
plt.show()
ax2.figure.savefig("intermediate_values.png")

# Plot parameter importances
ax3 = plot_param_importances(study)
plt.show()
ax3.figure.savefig("param_importances.png")
#%% md
# # Training optimized
#%%
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
#
# # Define the model
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
#
# # Load best hyperparameters (already defined earlier as best_hparams)
# training_args = TrainingArguments(
#     output_dir="./final_model",
#     learning_rate=best_hparams["learning_rate"],
#     per_device_train_batch_size=best_hparams["per_device_train_batch_size"],
#     weight_decay=best_hparams["weight_decay"],
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     logging_strategy="epoch",
#     num_train_epochs=best_hparams["num_train_epochs"],
#     warmup_steps=best_hparams["warmup_steps"],
#     report_to="wandb",
#     run_name="final_run_with_best_hparams",
# )
#
# # Create Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["valid"],
#     processing_class=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )
# from transformers import EarlyStoppingCallback
#
# trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
# # Train
# trainer.train()
#
# # Save the model
# trainer.save_model("./final_model")
#%%
# from matplotlib import pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
#
# # Generate predictions
# predictions = trainer.predict(tokenized_datasets["test"])
# predicted_labels = predictions.predictions.argmax(axis=-1)
#
# # Classification report
# print(classification_report(tokenized_datasets["test"]["label"], predicted_labels))
#
# # Confusion matrix
# cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nocb", "gender","ethnicity", "religion", "age","other"])
# disp.plot(cmap="Blues")  # Optional: set a color map
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=300)  # You can change the name or dpi as needed
# plt.close()  # Close the plot to free memory if you're in a loop
#%%
# # Inspect misclassified samples
# for idx, (pred, label) in enumerate(zip(predicted_labels, tokenized_datasets["test"]["label"])):
#     if pred != label:
#         print(f"Index: {idx}, Predicted: {pred}, Actual: {label}")
#         print(tokenized_datasets["test"][idx]["text"])
#%%
tokenizer.save_pretrained("./final_model")
print("Model saved successfully!")