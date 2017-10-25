import numpy as np
from string import ascii_lowercase, ascii_uppercase, digits


ACCEPTED_CHARACTERS = list(ascii_uppercase + ascii_lowercase + digits + "#$%&*()-_+=[]|:;\"\'<,>.?/")
NUM_ACCEPTED_CHARACTERS = len(ACCEPTED_CHARACTERS)
MAXIMUM_WORD_LENGTH = 210
GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"


char_index_map = {}
for i, c in enumerate(ACCEPTED_CHARACTERS):
	char_index_map[c] = i


def get_feature_vector_for_word(word, maximum_word_length):
	vector = np.zeros((1, NUM_ACCEPTED_CHARACTERS*maximum_word_length), dtype=bool)
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
					parts = token.split("_")
					word = parts[0]
					tag = parts[1]
					words_to_label[word] = get_label_for_tag(tag)
			else:
				break
		return words_to_label

def get_data_and_labels_from_corpus(corpus_path, maximum_word_length):
	word_to_label = extract_unique_words_and_labels(corpus_path)
	feature_vectors = []
	labels = []
	for word, label in word_to_label.iteritems():
		feature_vectors.append(get_feature_vector_for_word(word))
		labels.append(label)

	n = len(labels)
	d = NUM_ACCEPTED_CHARACTERS*maximum_word_length
	data_matrix = np.zeros((n, d), dtype=bool)
	labels_vector = np.array(labels, dtype=bool)
	for i in range(n):
		data_matrix[i:i+1, :] = feature_vectors[i]
	return data_matrix, labels_vector
