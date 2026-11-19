import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def k_nearest_neighbors(x, y, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x, y)
    predict_value = 2
    predictions = model.predict([[predict_value]])
    print("Predicted value for hours study = ", predict_value, "is", predictions)

    distances, indices = model.kneighbors([[predict_value]])

    print("Indices:", indices)
    print("Distances:", distances)
    