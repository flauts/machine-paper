#%%
import optuna
from datasets import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from joblib import load

SEED = 42
np.random.seed(SEED)


train_ds = load("../../feature_extraction/tfidf/train_tfidf.joblib")
valid_ds = load("../../feature_extraction/tfidf/valid_tfidf.joblib")
test_ds = load("../../feature_extraction/tfidf/test_tfidf.joblib")
train_label = load("../../feature_extraction/tfidf/train_labels.joblib")
valid_label = load("../../feature_extraction/tfidf/valid_labels.joblib")
test_label = load("../../feature_extraction/tfidf/test_labels.joblib")
# Convert HF Dataset columns to NumPy arrays
X_train = train_ds
y_train = np.array(train_label)
X_valid = (valid_ds)
y_valid = np.array(valid_label)
X_test = (test_ds)
y_test = np.array(test_label)

#%%

# ------------------------------
# Objective function for Optuna
# ------------------------------
X_train = X_train.toarray()  # Convertir matriz dispersa a densa
X_valid = X_valid.toarray()
y_train = np.array(y_train)
y_valid = np.array(y_valid)

def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)  # 1=Manhattan, 2=Euclidean

    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)

    # F1 macro is good for multiclass
    f1 = f1_score(y_valid, preds, average="macro")
    return f1

#%%
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30)

print("Best hyperparameters:", study.best_params)
print("Best validation F1:", study.best_value)

best_params = study.best_params
final_clf = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    weights=best_params["weights"],
    p=best_params["p"],
    n_jobs=-1
)
final_clf.fit(np.vstack((X_train, X_valid)), np.hstack((y_train, y_valid)))

test_preds = final_clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro-F1: {test_f1:.4f}")

from joblib import dump, load

# Save the trained model
dump(final_clf, "knn_tfidf_model.joblib")

# Later, to load the model
# loaded_model = load("knn_tfidf_model.joblib")
# preds = loaded_model.predict(X_test)
