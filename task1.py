from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

class NB_task1():

    def __init__(self):
        self.clf = None
        # pass

    def fit(self, X_train_tfidf, train_labels):
        self.clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, train_labels)
        # return clf

    def predict(self, X_test_tfidf):
        predicted = self.clf.predict(X_test_tfidf)
        return predicted

    def evaluate(self, train_labels, predicted):
        return accuracy_score(train_labels, predicted)

    def execute_task1(self):
        twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
        tfidf_vectorizer = TfidfVectorizer()

        X_train_tfidf = tfidf_vectorizer.fit_transform(twenty_train.data)
        clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, twenty_train.target)
        twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

        X_test_tfidf = tfidf_vectorizer.transform(twenty_test.data)
        print(X_test_tfidf.shape)

        predicted = clf.predict(X_test_tfidf)
        predicted

        accuracy = (np.sum(predicted == twenty_test.target) / len(predicted))*100
        print("Accuracy = ", accuracy)
        return accuracy

if __name__ == "__main__":

    obj = NB_task1()
    acc = obj.execute_task1()
