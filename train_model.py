from sklearn.model_selection import train_test_split
from ml.model import train_model, inference, compute_model_metrics, save_model, performance_on_categorical_slice
import pandas as pd

columns = [
    "age", "workclass", "fnlgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "salary"
]

data = pd.read_csv("data.csv", header = None, names = columns, skipinitialspace=True)

data["salary"] = data["salary"].astype(str).str.strip()
data["salary"] = data["salary"].apply(lambda x: 1 if x == ">50K" else 0)


data = data.replace("?", None).dropna()

X = pd.get_dummies(data.drop("salary", axis = 1))
y = data["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = train_model(X_train, y_train)

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("metrics: ", precision, recall, fbeta)

save_model(model, "model.pkl")
results = performance_on_categorical_slice(X_test, y_test, model, X_test.columns[0])
with open("slice_output.txt", "w") as f:
    for key, val in results.items():
        f.write(f"{key}: {val}\n")