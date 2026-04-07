Project Overview

In this project, a predictive model will be developed and deployed that determines if a person earns more than $50,000 annually using their demographic and work information. The predictive model will be accessible via a FastAPI-based REST API.

ML Model
Algorithm: Logistic Regression
Dataset: Adult Census Income Dataset
Task: Binary classification (>50K vs <=50K)

Instructions:
Before you try to run the project, first run the following commando to avoid dependencies problems:
- python -m pip install -r requirements.txt

To train the model:
python train_model.py

This will: Train the model, save model.pkl and generate slice_output.txt

To run the API:
python -m uvicorn main:app --reload

API will be available at:

http://127.0.0.1:8000

Finally: 
Test the API
python local_api.py
