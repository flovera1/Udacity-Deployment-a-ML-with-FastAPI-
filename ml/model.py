import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter = 6000, class_weight = 'balanced')
    model.fit(X_train, y_train)
    return model

def inference(model, X):
    return model.predict(X)

def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta = 1, zero_division=1)
    return precision, recall, fbeta

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def performance_on_categorical_slice(X, y, model, feature_name):
    results = {}
    for val in X[feature_name].unique():
        idx = X[feature_name] == val
        preds = inference(model, X[idx])
        precision, recall, fbeta = compute_model_metrics(y[idx], preds)
        results[val] = (precision, recall, fbeta)

    return results

