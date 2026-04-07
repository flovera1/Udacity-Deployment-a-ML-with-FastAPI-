import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def inference(model, X):
    return model.predict(X)


def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def performance_on_categorical_slice(X, y, model, feature_name, encoder, model_columns):
    results = []

    for val in X[feature_name].unique():
        idx = X[feature_name] == val

        X_slice = X[idx]
        y_slice = y[idx]

        if len(y_slice) == 0:
            continue

        # Encode slice like training
        X_slice_enc = encoder(X_slice)
        X_slice_enc = X_slice_enc.reindex(columns=model_columns, fill_value=0)

        preds = inference(model, X_slice_enc)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        results.append({
            "feature": feature_name,
            "value": val,
            "count": len(y_slice),
            "precision": precision,
            "recall": recall,
            "f1": fbeta
        })

    return results