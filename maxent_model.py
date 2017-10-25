from word_features import get_data_and_labels_from_corpus
from sklearn.linear_model import LogisticRegression


TRAIN_DATA_CORPUS = "data/train.tag"
DEV_DATA_CORPUS = "data/dev.tag"


train_d, train_l = get_data_and_labels_from_corpus(TRAIN_DATA_CORPUS)

logistic = LogisticRegression()
logistic.fit(train_d, train_l)


dev_d, dev_l = get_data_and_labels_from_corpus(DEV_DATA_CORPUS)
print "Score: ", logistic.score(dev_d, dev_l)