
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, vstack, hstack



class NB_using_EM_task2():
    def __init__(self):
        pass

    def perform_EM(self, labelled_X_tfidf, labelled_y, unlabelled_X_tfidf):
        multinomial_clf = MultinomialNB(alpha=0.01)  # Defacto ALWAYS use this
        multinomial_clf.fit(labelled_X_tfidf, labelled_y)
        class_prob = multinomial_clf.class_log_prior_
        feature_prob = multinomial_clf.feature_log_prob_
        old_loss = 0
        loss = -1 * (self.calculate_unlabeled_loss(class_prob, feature_prob, unlabelled_X_tfidf) + self.calculate_labelled_loss(
            class_prob, feature_prob, labelled_X_tfidf, labelled_y))
        diff = (loss - old_loss)
        threshold = np.exp(-10)
        while (abs(diff) > threshold):
            old_loss = loss
            # E-step
            pred_labels = multinomial_clf.predict(unlabelled_X_tfidf)

            # Combining the data
            X_total = vstack([labelled_X_tfidf, unlabelled_X_tfidf])
            Y_total = np.concatenate((labelled_y, pred_labels), axis=0)
            X_total.shape, Y_total.shape

            # M-step
            multinomial_clf.fit(X_total, Y_total)

            class_prob = multinomial_clf.class_log_prior_
            feature_prob = multinomial_clf.feature_log_prob_
            loss = -1 * (self.calculate_unlabeled_loss(class_prob, feature_prob,
                                                  unlabelled_X_tfidf) + self.calculate_labelled_loss(class_prob,
                                                                                                feature_prob,
                                                                                                labelled_X_tfidf,
                                                                                                labelled_y))

            diff = old_loss - loss
            print("diff", diff)
        return multinomial_clf

    def calculate_unlabeled_loss(self, class_prob, feature_prob, unlabelled_data_tfidf):
        """
        unlabeled_data_one_hot_matrix: Assuming this to be ndarray
        """
        n = unlabelled_data_tfidf.shape[0]

        p_xi_cj = np.sum(feature_prob, axis=1, keepdims=True)
        class_prob = class_prob.reshape(class_prob.shape[0], 1)

        sum_loss = class_prob + p_xi_cj
        sum_for_all_classes = np.sum(sum_loss, axis=0, keepdims=True)
        sum_for_all_classes = sum_for_all_classes[0]
        total_loss = n * sum_for_all_classes
        return total_loss

    def calculate_labelled_loss(self, class_prob, feature_prob, labelled_data_tfidf, labelled_labels):
        n_labelled = labelled_data_tfidf.shape[0]
        loss = 0
        sum_of_feature_log_probs = np.sum(feature_prob, axis=1, keepdims=True)
        for i in range(n_labelled):
            c_index = labelled_labels[i]
            prob_for_c = class_prob[c_index]
            feature_value_for_c = sum_of_feature_log_probs[c_index]
            loss += (prob_for_c + feature_value_for_c)
        return loss



if __name__ == "__main__":

    obj = NB_using_EM_task2()
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    labelled_X, unlabelled_X, labelled_y, _ = train_test_split(twenty_train.data, twenty_train.target, train_size=0.2)

    # Finding the TF-IDF from the whole data for labelled and unlabelled data (Feature extraction step)
    # Step-2
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(twenty_train.data)
    labelled_X_tfidf = tfidf_vectorizer.transform(labelled_X)
    unlabelled_X_tfidf = tfidf_vectorizer.transform(unlabelled_X)

    em_trained_clf = obj.perform_EM(labelled_X_tfidf, labelled_y, unlabelled_X_tfidf)

    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    X_test_tfidf = tfidf_vectorizer.transform(twenty_test.data)
    predicted = em_trained_clf.predict(X_test_tfidf)

    # print(accuracy_score(twenty_test.target, predicted))
    print("Accuracy = ", (np.sum(predicted == twenty_test.target) / len(predicted)) * 100)