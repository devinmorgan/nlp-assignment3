import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt1:
    def __init__(self):
        self.logistic = LogisticRegression()

    @staticmethod
    def get_feature_vector_for_word(word):
        vector = np.zeros((1, wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH), dtype=bool)
        for i, c in enumerate(word):
            char_index_offset = i * wf.NUM_ACCEPTED_CHARACTERS
            char_value_offset = wf.char_index_map[c]
            feature_index = char_index_offset + char_value_offset
            vector[:, feature_index] = 1
        return vector

    @staticmethod
    def extract_words_and_labels(corpus_path):
        with open(corpus_path) as f:
            words_to_label = {}
            words_list = []
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    for token in text.split(" "):
                        parts = token.split("_")
                        word = parts[0]
                        tag = parts[1]
                        words_to_label[word] = wf.get_label_for_tag(tag)
                        words_list.append(word)
                else:
                    break
            return words_to_label, words_list

    @staticmethod
    def extract_data_and_labels(corpus_path):
        words_to_label, _ = MaxEnt1.extract_words_and_labels(corpus_path)
        feature_vectors = []
        labels = []
        for word, label in words_to_label.iteritems():
            feature_vectors.append(MaxEnt1.get_feature_vector_for_word(word))
            labels.append(label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH
        data_matrix = np.zeros((n, d), dtype=bool)
        labels_vector = np.array(labels, dtype=bool)
        for i in range(n):
            data_matrix[i:i + 1, :] = feature_vectors[i]
        return data_matrix, labels_vector

    def train(self, training_data_file):
        train_d, train_l = MaxEnt1.extract_data_and_labels(training_data_file)
        self.logistic.fit(train_d, train_l)

    def tag_text(self, text):
        tokens = text.strip().split(" ")
        new_tokens = []
        for t in tokens:
            word = t.split("_")[0]
            feature_vector = MaxEnt1.get_feature_vector_for_word(word)
            prediction = self.logistic.predict(feature_vector)
            tag = wf.NON_GENE_TAG
            if prediction == 1:
                tag = wf.GENE_TAG
            new_token = word + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)
