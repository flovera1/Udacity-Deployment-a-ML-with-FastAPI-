import numpy as np
from ml.model import compute_model_metrics, inference, train_model

def test_metrics():
    y = np.array([1, 0, 1])
    preds = np.array([1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0

def test_inference_shape():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)

def test_model_training():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = train_model(X, y)
    assert model is not None

