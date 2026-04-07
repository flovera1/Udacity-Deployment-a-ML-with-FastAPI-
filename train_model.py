import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    performance_on_categorical_slice
)

# Column names for Adult dataset
columns = [
    "age", "workclass", "fnlgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "salary"
]

# Load data
data = pd.read_csv("data.csv", header=None, names=columns, skipinitialspace=True)

# Clean missing values
data = data.replace("?", pd.NA).dropna()

# Clean target
data["salary"] = data["salary"].astype(str).str.strip().str.upper()
data["salary"] = data["salary"].apply(lambda x: 1 if x == ">50K" else 0)

# Split features
X = data.drop("salary", axis=1)
y = data["salary"]

# Encode features
encoder = pd.get_dummies(X)
model_columns = encoder.columns

# Train/test split (IMPORTANT: keep both encoded + raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train_enc = pd.get_dummies(X_train)
X_test_enc = pd.get_dummies(X_test)

# Align columns
X_train_enc = X_train_enc.reindex(columns=model_columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=model_columns, fill_value=0)

# Train model
model = train_model(X_train_enc, y_train)

# Evaluate
preds = inference(model, X_test_enc)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Metrics:", precision, recall, fbeta)

# Save model
save_model(model, "model.pkl")

# Slice analysis
categorical_features = ["workclass", "education"]

with open("slice_output.txt", "w") as f:
    for feature in categorical_features:
        results = performance_on_categorical_slice(
            X_test,
            y_test,
            model,
            feature,
            pd.get_dummies,
            model_columns
        )

        for res in results:
            f.write(
                f"{res['feature']}: {res['value']}, Count: {res['count']}\n"
            )
            f.write(
                f"Precision: {res['precision']:.4f} | "
                f"Recall: {res['recall']:.4f} | "
                f"F1: {res['f1']:.4f}\n\n"
            )


