import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def logistic_regression(x, y):
    model = LogisticRegression()
    model.fit(x,y)

    predict_hour = 1
    predict_probabilities = model.predict_proba([[predict_hour]])
    predictions = model.predict([[predict_hour]])
    print("Probabilities: ", predict_probabilities)
    print("Predicted value", predictions)