import numpy as np
from string import ascii_lowercase, ascii_uppercase, digits


ACCEPTED_CHARACTERS = list(ascii_uppercase + ascii_lowercase + digits + "#$%&*()-_+=[]|:;\"\'<,>.?/")
NUM_ACCEPTED_CHARACTERS = len(ACCEPTED_CHARACTERS)
MAXIMUM_WORD_LENGTH = 80
GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"



char_index_map = {}
for i, c in enumerate(ACCEPTED_CHARACTERS):
	char_index_map[c] = i


def get_feature_vector_for_word(word):
	vector = np.zeros((1, NUM_ACCEPTED_CHARACTERS*MAXIMUM_WORD_LENGTH))
	for i, c in enumerate(word):
		char_index_offset = i*NUM_ACCEPTED_CHARACTERS
		char_value_offset = char_index_map[c]
		feature_index = char_index_offset + char_value_offset
		vector[:, feature_index] = 1
	return vector


def get_label_for_tag(tag):
	if tag == TAG_LABEL:
		return 0
	if tag == GENE_1_LABEL:
		return 1
	if tag == GENE_2_LABEL:
		return 1


def extract_unique_words_and_labels(corpus_path):
	with open(corpus_path) as f:
		words_to_label = {}
		while True:
			id = f.readline()
			text = f.readline().strip()
			if text:
				for token in text.split(" "):
					data = token.split("_")
					word = data[0]
					tag = data[1]
					words_to_label[word] = tag
			else:
				break
		return words_to_label

def get_data_and_labels_from_corpus(corpus_path):
	word_to_label = extract_unique_words_and_labels(corpus_path)
	feature_vectors = []
	labels = []
	for word, label in word_to_label.iteritems():
		feature_vectors.append(get_feature_vector_for_word(word))
		labels.append(label)

	n = len(labels)
	d = NUM_ACCEPTED_CHARACTERS*MAXIMUM_WORD_LENGTH
	data_matrix = np.zeros((n, d))
	labels_vector = np.array(labels)
	for i in range(n):
		data_matrix[i:i+1, :] = feature_vectors[i]
	return data_matrix, labels_vector
