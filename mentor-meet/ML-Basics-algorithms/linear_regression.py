import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x,y)

    print("Coefficients: ", model.coef_)
    print("Intercept: ", model.intercept_)

    predictions = model.predict([[7]])
    print("Predicted value", predictions)

    plt.scatter(x, y, color='blue')
    plt.plot(x, model.predict(x), color='red')

    plt.title('Linear Regression')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks Scored')
    plt.show()