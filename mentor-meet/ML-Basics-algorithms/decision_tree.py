import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def decision_tree(x, y):
    model = DecisionTreeClassifier()
    model.fit(x, y)
    print("Feature Importances: ", model.feature_importances_)
    plt.figure(figsize=(12,6))
    plot_tree(model, filled=True, )
    plt.title('Decision Tree Classifier')
    plt.show()