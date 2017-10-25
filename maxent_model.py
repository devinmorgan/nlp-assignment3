from word_features import get_data_and_labels_from_corpus, get_feature_vector_for_word
from sklearn.linear_model import LogisticRegression


TRAIN_DATA_CORPUS = "data/train.tag"
DEV_DATA_CORPUS = "data/dev.tag"
TEST_DATA_CORPUS = "data/test.tag"

MODEL_1_OUTPUT_FILE = "output/output_test1.tag"
MODEL_2_OUTPUT_FILE = "output/output_test2.tag"
MODEL_3_OUTPUT_FILE = "output/output_test3.tag"


def get_trained_model1():
	train_d, train_l = get_data_and_labels_from_corpus(TRAIN_DATA_CORPUS)
	logistic = LogisticRegression()
	logistic.fit(train_d, train_l)
	return logistic


def predict_text_with_model(text, model):
	tokens = text.strip().split(" ")
	new_tokens = []
	for t in tokens:
		parts = t.split("_")
		word = parts[0]
		feature_vector = get_feature_vector_for_word(word)
		pred = model.predict(feature_vector)

		tag = parts[1]
		if pred == 0:
			tag = "TAG"
		elif pred == 1:
			tag = "GENE1"
		new_token = word + "_" + tag
		new_tokens.append(new_token)
	return " ".join(new_tokens)



def write_output_to_file(output, file_name):
	with open(file_name, 'w') as f:
		for line in output:
			f.write(line + "\n")


def predict_test_data(model, output_file):
	with open(DEV_DATA_CORPUS) as f:
		output = []
		while True:
			id = f.readline().strip()
			output.append(id)
			text = f.readline().strip()
			if text:
				predicted_text = predict_text_with_model(text, model)
				output.append(predicted_text)
			else:
				break
		write_output_to_file(output, output_file)

m1 = get_trained_model1()
predict_test_data(m1, MODEL_1_OUTPUT_FILE)
# logistic = get_trained_model1()
# dev_d, dev_l = get_data_and_labels_from_corpus(DEV_DATA_CORPUS)
# print "Score: ", logistic.score(dev_d, dev_l)