# Model Card – Adult Income Prediction
## Model Details
In this project I used logistic regression to develop a model that predicts if an
individua earns more than 50K dollars annually based on various factors such as demographics
and employment. This model was bult using the scikitlearn library and was eventually deployed
as part of the FastAPI.
## Intended Use
This is a model that is meant for education purposes only.
## Data
The training process was done using the Adult Census Income data set, which comprises both numeric and categorical input values. Among those variables that we have: age, education, profession, marriage, and weekly working hours.
Preprocessing actions included:
- Deletion of missing values ("?")
- One-hot encoding of categorical values
- Transformation of the target variable (`salary`):
  -- 1 – income > $50K
  -- 0 – income <= $50K
## Metric
- Precision: 0.558
- Recall: 0.816
- F1 Score: 0.663
### Interpretation
These metrics mean that the classifier has identified almost all people earning over $50K. Nevertheless, low values of the precision measure mean that there may be mistakes in predicting those people.
Then, the classifier prefers to predict high-income individuals but makes more errors in doing so. 
## Ethical Considerations
Since the data set consists of characteristics like race and gender, there exists a possibility that the machine might end up learning some of the biases prevalent in society.
## Limitations
There are critical and several limitations to this model:

- The dataset is imbalanced
- Logistic Regression assumes linear relationships and it won't capture more complex patterns
- One-hot encoding may not handle unseen categories well
- The model has not been evaluated across different demographic groups for fairness
## Summary
This assignment illustrates a full machine learning process, including data preparation and deployment. Although the model performs adequately well, its primary purpose is for educational purposes and not for practical use.
 