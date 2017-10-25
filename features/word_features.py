import numpy as np
from string import ascii_lowercase, ascii_uppercase, digits

ACCEPTED_CHARACTERS = list(ascii_uppercase + ascii_lowercase + digits + "!@#$%^&*()-_+={[}]|\\:;\"\'<,>.?/")
NUM_ACCEPTED_CHARACTERS = len(ACCEPTED_CHARACTERS)
MAXIMUM_WORD_LENGTH = 100

char_index_map = {}
for i, c in enumerate(ACCEPTED_CHARACTERS):
	char_index_map[c] = i


def get_feature_for_word(word):
	vector = np.zeros((NUM_ACCEPTED_CHARACTERS*MAXIMUM_WORD_LENGTH, 1))
	for i, c in enumerate(word):
		char_index_offset = i*NUM_ACCEPTED_CHARACTERS
		char_value_offset = char_index_map[c]
		feature_index = char_index_offset + char_value_offset
		vector[feature_index, :]= 1
	return vector


