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
for j, ch in enumerate(ACCEPTED_CHARACTERS):
    char_index_map[ch] = j


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
