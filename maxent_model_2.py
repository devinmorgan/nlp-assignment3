import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt2:
    def __init__(self):
        self.logistic = LogisticRegression()

    @staticmethod
    def get_feature_vector_for_word(word):
        pass

    @staticmethod
    def extract_data_and_labels(corpus_path, context_word_size):
        words_to_label, words_list = wf.extract_words_and_labels(corpus_path)
        words_list = [""] * (context_word_size - 1) + words_list
        feature_vectors = []
        labels = []
        for i in xrange(context_word_size - 1, len(words_list)):
            prev_prev_word = words_list[i - 2]
            prev_word = words_list[i - 1]
            word = words_list[i]
            trio = (
                MaxEnt2.get_feature_vector_for_word(prev_prev_word),
                MaxEnt2.get_feature_vector_for_word(prev_word),
                MaxEnt2.get_feature_vector_for_word(word)
            )
            feature_vector = np.concatenate(trio, axis=1)
            feature_vectors.append(feature_vector)
            label = words_to_label[word]
            labels.append(label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH * context_word_size
        data_matrix = np.zeros((n, d), dtype=bool)
        labels_vector = np.array(labels, dtype=bool)
        for i in range(n):
            data_matrix[i:i + 1, :] = feature_vectors[i]
        return data_matrix, labels_vector

    def train(self, training_data_file):
        train_d, train_l = MaxEnt2.extract_data_and_labels(training_data_file)
        self.logistic.fit(train_d, train_l)

    def tag_text(self, text):
        tokens = text.strip().split(" ")
        new_tokens = []
        for t in tokens:
            parts = t.split("_")
            word = parts[0]
            tag = parts[1]
            feature_vector = MaxEnt2.get_feature_vector_for_word(word)
            prediction = self.logistic.predict(feature_vector)
            if prediction == 0:
                tag = wf.NON_GENE_TAG
            elif prediction == 1:
                tag = wf.GENE_TAG
            new_token = word + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)
