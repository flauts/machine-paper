#%%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

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
tokenizer = AutoTokenizer.from_pretrained("../../models/distilroberta_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("../../models/distilroberta_finetuned")
#%%

def tokenize_function(examples): #padding false becase of data collator later
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=200)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
print(tokenized_datasets["train"][0])
#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#%%

trainer = Trainer(
    model = model,
    data_collator = data_collator,
    eval_dataset = tokenized_datasets["test"]
)
results = trainer.evaluate()
print(results)