import numpy as np
from decision_tree import decision_tree
import naive_bayes
import linear_regression
import logistics_regression
import k_nearest

## Linear Regression
# x: study hours, y: marks scored
x = np.array([
    [1], [2], [2.5], [3], [3.5],
    [4], [4.5], [5], [5.5], [6],
    [6.5], [7], [8], [9], [10],
])
y = np.array([30, 37, 42, 45, 51, 56, 61, 65, 69, 74, 78, 82, 88, 93, 97])
linear_regression.linear_regression(x, y)


# ## Logistic Regression
# x = np.array([[1], [2], [3], [4], [5]])
# y = np.array(["Pass", "Pass", "Fail", "Pass", "Pass"])
# logistics_regression.logistic_regression(x, y)


# ## Decision Tree
# X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
# y = np.array(["Fail", "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass"])
# decision_tree(X, y)

# ### K-Nearest Neighbors
# x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# y = np.array(["Fail", "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass"])
# k_nearest.k_nearest_neighbors(x, y, n_neighbors=3)


## Naive Bayes
##Gaussian Naive Bayes
# # Training data
# # [Height, Weight]
# X = np.array([[150, 50],[155, 52],[160, 55],[165, 60],[170, 65],[175, 70],[180, 75], [185, 80]])

# # 0 = Short
# # 1 = Tall
# y = np.array([1,0,0,0,1,0,1,1])
# naive_bayes.Gassian_Naive_Bayes(X, y)

# # multinomial Naive Bayes
# messages = ["free money","free prize","win money","meeting today","project meeting","project update"]

# # 1 = Spam
# # 0 = Not Spam
# y = ["Spam", "Spam", "Spam", "Not Spam", "Not Spam", "Not Spam"]
# naive_bayes.multinomial_Naive_Bayes(messages, y)
