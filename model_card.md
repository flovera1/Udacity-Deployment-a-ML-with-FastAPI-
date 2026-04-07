# Model card
## Model details
This project uses a Logistic Regression model trained on the Adult Census Income (public available).
## Intent
Binary classification where demographic predic income category (>50K or <=50K; the capital K matters)
## Metrics
The model was evaluated under these metrics:
- Precision: 0.558  
- Recall: 0.816  
- F1: 0.663
What it means is: having high recall but lower precision, so the model is better at identifying positive cases than avoiding false positives.
## Limitatons
Depends on dataset quality.