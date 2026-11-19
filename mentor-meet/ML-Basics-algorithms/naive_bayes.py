import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def Gassian_Naive_Bayes(X, y):
    print("Gaassian Naive Bayes")
    model = GaussianNB()
    model.fit(X, y)
    predict_value = [[160, 55]]
    predictions = model.predict(predict_value)
    print("Predicted value for Height = ", predict_value[0][0], "and Weight = ", predict_value[0][1], "is", predictions)

def multinomial_Naive_Bayes(messages, y):
    print("Multinomial Naive Bayes")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages)
    print(X.toarray())
    model = MultinomialNB()
    model.fit(X, y)
    predict_message = ["You won free money - touch the link to claim it"]
    predict_message_vectorized = vectorizer.transform(predict_message)
    predictions = model.predict(predict_message_vectorized)
    print("Predicted value for message '", predict_message[0], "' is", predictions)