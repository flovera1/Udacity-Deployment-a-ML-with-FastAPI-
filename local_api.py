import requests

r = requests.get("http://127.0.0.1:8000/")
print(r.status_code, r.json())

data = {"feature1": 1, "feature2": 2}

r = requests.post("http://127.0.0.1:8000/predict", json = data)
print(r.status_code, r.json())

