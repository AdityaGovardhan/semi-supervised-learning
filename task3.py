from nltk.corpus import reuters
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, vstack, hstack
from sklearn.metrics import accuracy_score



class Task3():

    def __init__(self):
        pass

    def represent(self, documents):
        train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
        test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

        train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
        test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

        # Tokenization
        vectorizer = TfidfVectorizer()

        # Learn and transform train documents
        vectorised_train_documents = vectorizer.fit_transform(train_docs)
        vectorised_test_documents = vectorizer.transform(test_docs)

        return vectorised_train_documents, vectorised_test_documents

    def get_10_pos_docs(self, labelled_y, categs_sorted_by_doc_num):
        cat_docs = {}
        # all_copy.deepcopy(labelled_y)
        visited_docs = []
        for categ in categs_sorted_by_doc_num:
            for i, j in enumerate(labelled_y):
                if((categ in j) and (i not in visited_docs)):
                    if(categ in cat_docs):
                        cat_docs[categ].append(i)
                    else:
                        cat_docs[categ] = [i]
                    visited_docs.append(i)
                    if(len(cat_docs.get(categ)) == 10):
                        break
        return cat_docs

    def get_40_neg_docs(self, labelled_y, categs_sorted_by_doc_num):
        neg_cat_docs = {}
        # all_copy.deepcopy(labelled_y)
        visited_docs1 = []
        for categ in categs_sorted_by_doc_num:
            for i, j in enumerate(labelled_y):
                if ((categ not in j) and (i not in visited_docs1)):
                    if (categ in neg_cat_docs):
                        neg_cat_docs[categ].append(i)
                    else:
                        neg_cat_docs[categ] = [i]
                    visited_docs1.append(i)
                    if (len(neg_cat_docs.get(categ)) == 40):
                        break
        return neg_cat_docs


    def get_positive_train_data(self, pos_cat_docs, labelled_X):
        positive_train_data = {}
        for each_cat in pos_cat_docs.keys():
            list_of_doc_indexes = pos_cat_docs.get(each_cat)
            #     cat_doc_vec[each_cat] =
            for doc_index in list_of_doc_indexes:
                if (each_cat not in positive_train_data):
                    positive_train_data[each_cat] = labelled_X[doc_index]
                else:
                    positive_train_data[each_cat] = vstack([positive_train_data[each_cat], labelled_X[doc_index]])

        return positive_train_data

    def get_neg_train_data(self, neg_cat_docs, labelled_X):
        neg_train_data = {}
        for each_cat in neg_cat_docs.keys():
            list_of_doc_indexes = neg_cat_docs.get(each_cat)
            #     cat_doc_vec[each_cat] =
            for doc_index in list_of_doc_indexes:
                if (each_cat not in neg_train_data):
                    neg_train_data[each_cat] = labelled_X[doc_index]
                else:
                    neg_train_data[each_cat] = vstack([neg_train_data[each_cat], labelled_X[doc_index]])

        return neg_train_data

    def get_accuracies(self, categs_sorted_by_doc_num):
        clfs = [0] * 10
        accs = {}
        for i, cat in enumerate(categs_sorted_by_doc_num):
            train_data_for_each_cat = vstack([positive_train_data[cat], neg_train_data[cat]])
            train_labels_for_each_cat = np.concatenate((np.ones(10), np.zeros(40)))

            clfs[i] = MultinomialNB(alpha=0.01)
            clfs[i].fit(train_data_for_each_cat, train_labels_for_each_cat)

            a = clfs[i].predict(test_data)

            corr_labels = [1 if cat in x else 0 for x in test_labels]

            accs[cat] = accuracy_score(corr_labels, a)
        return accs

    def get_most_populous_categs(self):
        cats = reuters.categories()

        categ_dict = {}

        total_multi = 0
        for c in cats:
            lcat = len(reuters.paras(categories=[c]))
            total_multi += lcat
            categ_dict[c] = lcat
        most_populous_categs = sorted(categ_dict.items(), key=itemgetter(1), reverse=True)
        # getting top 10
        top_10_poplous_categs = most_populous_categs[:10]

        top_10_poplous_categs
        top_10_poplous_categs_names = [i[0] for i in top_10_poplous_categs]
        return top_10_poplous_categs_names

if __name__ == "__main__":
    task3_obj = Task3()
    train_data, test_data = task3_obj.represent(reuters.fileids())
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
    train_labels = [reuters.categories(doc_id) for doc_id in train_docs_id]
    test_labels = [reuters.categories(doc_id) for doc_id in test_docs_id]
    ratio = 7000/9603
    labelled_X, unlabelled_X, labelled_y, _ = train_test_split(train_data, train_labels, train_size=1 - ratio)
    top_10_poplous_categs_names = task3_obj.get_most_populous_categs()

    cat_docs = {}
    for categ in top_10_poplous_categs_names:
        cat_docs[categ] = [i for i, j in enumerate(labelled_y) if categ in j]

    pos_cat_docs = task3_obj.get_10_pos_docs(labelled_y, top_10_poplous_categs_names)
    neg_cat_docs = task3_obj.get_40_neg_docs(labelled_y, top_10_poplous_categs_names)

    positive_train_data = task3_obj.get_positive_train_data(pos_cat_docs, labelled_X)
    neg_train_data = task3_obj.get_neg_train_data(pos_cat_docs, labelled_X)

    Accuracy = task3_obj.get_accuracies(top_10_poplous_categs_names)

    print(Accuracy)

