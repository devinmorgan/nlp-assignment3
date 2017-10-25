from word_features import get_data_and_labels_from_corpus
from sklearn.linear_model import LogisticRegression


TRAIN_DATA_CORPUS = "data/train.tag"
DEV_DATA_CORPUS = "data/dev.tag"
TEST_DATA_CORPUS = "data/test.tag"

def get_trained_model1():
	train_d, train_l = get_data_and_labels_from_corpus(TRAIN_DATA_CORPUS)
	logistic = LogisticRegression()
	logistic.fit(train_d, train_l)
	return logistic


def predict_text_with_model(text, model):
	pass


def write_output_to_file(output):
	pass


def predict_test_data(model):
	with open(TEST_DATA_CORPUS) as f:
		output = []
		while True:
			id = f.readline()
			output.append(id)
			text = f.readline().strip()
			if text:
				predicted_text = predict_text_with_model(text, model)
				output.append(predicted_text)
			else:
				break
		write_output_to_file(output)

# dev_d, dev_l = get_data_and_labels_from_corpus(DEV_DATA_CORPUS)
# print "Score: ", logistic.score(dev_d, dev_l)