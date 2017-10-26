import numpy as np
from string import ascii_lowercase, ascii_uppercase, digits


ACCEPTED_CHARACTERS = list(ascii_uppercase + ascii_lowercase + digits + "#$%&*()-_+=[]|:;\"\'<,>.?/")
NUM_ACCEPTED_CHARACTERS = len(ACCEPTED_CHARACTERS)
MAXIMUM_WORD_LENGTH = 70

GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"

NON_GENE_TAG = "TAG"
GENE_TAG = "GENE1"


char_index_map = {}
for j, c in enumerate(ACCEPTED_CHARACTERS):
    char_index_map[c] = j


def get_feature_vector_for_word(word):
    vector = np.zeros((1, NUM_ACCEPTED_CHARACTERS*MAXIMUM_WORD_LENGTH), dtype=bool)
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


def extract_words_and_labels(corpus_path):
    with open(corpus_path) as f:
        words_to_label = {}
        words_list = []
        while True:
            id = f.readline()
            text = f.readline().strip()
            if text:
                for token in text.split(" "):
                    parts = token.split("_")
                    word = parts[0]
                    tag = parts[1]
                    words_to_label[word] = get_label_for_tag(tag)
                    words_list.append(word)
            else:
                break
        return words_to_label, words_list


