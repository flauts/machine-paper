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
#%% md
# # TF-IDF Feature Extraction
# We'll use scikit-learn's TfidfVectorizer to extract TF-IDF features from the text data.
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))

# Fit and transform the training data
X_train = tfidf.fit_transform(dataset["train"]["text"])
# Transform validation and test data
X_valid = tfidf.transform(dataset["valid"]["text"])
X_test = tfidf.transform(dataset["test"]["text"])

# Convert to arrays for convenience
y_train = dataset["train"]["label"]
y_valid = dataset["valid"]["label"]
y_test = dataset["test"]["label"]

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_valid.shape}")
print(f"Test features shape: {X_test.shape}")
#%%
import joblib

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# Save the TF-IDF features
joblib.dump(X_train, 'train_tfidf.joblib')
joblib.dump(X_valid, 'valid_tfidf.joblib')
joblib.dump(X_test, 'test_tfidf.joblib')
joblib.dump(y_train, 'train_labels.joblib')
joblib.dump(y_test,'test_labels.joblib')
joblib.dump(y_valid,'valid_labels.joblib')