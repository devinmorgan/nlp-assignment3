import numpy as np
from string import ascii_lowercase, ascii_uppercase, digits


ACCEPTED_CHARACTERS = list(ascii_uppercase + ascii_lowercase + digits + "!@#$%^&*()-_+={[}]|\\:;\"\'<,>.?/")
NUM_ACCEPTED_CHARACTERS = len(ACCEPTED_CHARACTERS)
MAXIMUM_WORD_LENGTH = 100
GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"
TRAINING_DATA_CORPUS = "../data/train.tag"


char_index_map = {}
for i, c in enumerate(ACCEPTED_CHARACTERS):
	char_index_map[c] = i


def get_feature_vector_for_word(word):
	vector = np.zeros((NUM_ACCEPTED_CHARACTERS*MAXIMUM_WORD_LENGTH, 1))
	for i, c in enumerate(word):
		char_index_offset = i*NUM_ACCEPTED_CHARACTERS
		char_value_offset = char_index_map[c]
		feature_index = char_index_offset + char_value_offset
		vector[feature_index, :]= 1
	return vector


def get_label_for_tag(tag):
	if tag == TAG_LABEL:
		return 0
	if tag == GENE_1_LABEL:
		return 1
	if tag == GENE_2_LABEL:
		return 1


def get_data_and_labels_from_corpus(corpus_path):
	with open(corpus_path) as f:
		feature_vectors = []
		labels = []
		count = 0
		while True:
			id = f.readline()
			text = f.readline().strip()
			count += 1
			for token in text.split(" "):
				data = token.split("_")
				word = data[0]
				tag = data[1]
				feature_vector = get_feature_vector_for_word(word)
				feature_vectors.append(feature_vector)
				label = get_label_for_tag(tag)
				labels.append(label)
			if not text:
				break

		n = len(labels)
		d = NUM_ACCEPTED_CHARACTERS
		data_matrix = np.zeros((d, n))
		labels_vector = np.array([labels])
		for i in range(n):
			data_matrix[:, i:i+1] = np.array(feature_vectors[i])
		return data_matrix, labels_vector

data, labels = get_data_and_labels_from_corpus(TRAINING_DATA_CORPUS)
print(data)
print(labels)