import os

DATA_DIR = "data"
TWENTY_NEWSGROUPS_DIR = DATA_DIR + "/" + "20newsgroups"
TWENTY_NEWSGROUPS_TRAINING_DIR = TWENTY_NEWSGROUPS_DIR + "/" + "20news-bydate-train"
TWENTY_NEWSGROUPS_TESTING_DIR = TWENTY_NEWSGROUPS_DIR + "/" + "20news-bydate-test"

class Twenty_Newsgroups:
  def __init__(self):
    pass

  # Returns a dictionary with keys as the collection name
  # and value as a list of {filename, filedata} dicts
  def get_all_training_articles(self):
    collectionwise_training_folders = os.listdir(TWENTY_NEWSGROUPS_TRAINING_DIR)

    collectionwise_training_dataset = dict()

    for sub_folder in collectionwise_training_folders:
      collection_folder = TWENTY_NEWSGROUPS_TRAINING_DIR + "/" + sub_folder
      collectionwise_training_dataset[sub_folder] = list()
      articles = os.listdir(collection_folder)
      for article in articles:
        val = collectionwise_training_dataset[sub_folder]
        f = open(collection_folder + "/" + article, 'r')
        article_data = f.readlines()
        val.append({"article": article, "file_data": article_data})
        collectionwise_training_dataset[sub_folder] = val
        f.close()

    return collectionwise_training_dataset

  # Returns a dictionary with single key as the requested collection name
  # and value as a list of {filename, filedata} dicts
  def get_all_training_articles_for_collection(self, collection_name):

    collectionwise_training_dataset = dict()

    collection_folder = TWENTY_NEWSGROUPS_TRAINING_DIR + "/" + collection_name
    collectionwise_training_dataset[collection_name] = list()
    articles = os.listdir(collection_folder)
    for article in articles:
      val = collectionwise_training_dataset[collection_name]
      f = open(collection_folder + "/" + article, 'r')
      article_data = f.readlines()
      val.append({"article": article, "file_data": article_data})
      collectionwise_training_dataset[collection_name] = val
      f.close()

    return collectionwise_training_dataset

  # Returns a dictionary with keys as the collection name
  # and value as a list of {filename, filedata} dicts
  def get_all_testing_articles(self):
    collectionwise_testing_folders = os.listdir(TWENTY_NEWSGROUPS_TESTING_DIR)

    collectionwise_testing_dataset = dict()

    for sub_folder in collectionwise_testing_folders:
      collection_folder = TWENTY_NEWSGROUPS_TESTING_DIR + "/" + sub_folder
      collectionwise_testing_dataset[sub_folder] = list()
      articles = os.listdir(collection_folder)
      for article in articles:
        val = collectionwise_testing_dataset[sub_folder]
        f = open(collection_folder + "/" + article, 'r')
        article_data = f.readlines()
        val.append({"article": article, "file_data": article_data})
        collectionwise_testing_dataset[sub_folder] = val
        f.close()

    return collectionwise_testing_dataset

  # Returns a dictionary with single key as the requested collection name
  # and value as a list of {filename, filedata} dicts
  def get_all_testing_articles_for_collection(self, collection_name):

    collectionwise_testing_dataset = dict()

    collection_folder = TWENTY_NEWSGROUPS_TESTING_DIR + "/" + collection_name
    collectionwise_testing_dataset[collection_name] = list()
    articles = os.listdir(collection_folder)
    for article in articles:
      val = collectionwise_testing_dataset[collection_name]
      f = open(collection_folder + "/" + article, 'r')
      article_data = f.readlines()
      val.append({"article": article, "file_data": article_data})
      collectionwise_testing_dataset[collection_name] = val
      f.close()

    return collectionwise_testing_dataset

# Test
if __name__ == "__main__":
  twenty_newsgroups = Twenty_Newsgroups()

  training_dataset = twenty_newsgroups.get_all_training_articles()
  print(training_dataset.keys())

  collection_name = "alt.atheism"
  training_dataset_for_colletion = twenty_newsgroups.get_all_training_articles_for_collection(collection_name)
  print(training_dataset_for_colletion.keys())
  print(len(training_dataset_for_colletion[collection_name]))