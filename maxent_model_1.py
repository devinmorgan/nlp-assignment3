import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt1:
    def __init__(self):
        self.logistic = LogisticRegression()

    @staticmethod
    def get_feature_vector_for_word(word):
        return wf.get_feature_vector_for_word(word)

    @staticmethod
    def extract_data_and_labels(corpus_path):
        words_to_label, _ = wf.extract_words_and_labels(corpus_path)
        feature_vectors = []
        labels = []
        for word, label in words_to_label.iteritems():
            feature_vectors.append(wf.get_feature_vector_for_word(word))
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
            parts = t.split("_")
            word = parts[0]
            tag = parts[1]
            feature_vector = MaxEnt1.get_feature_vector_for_word(word)
            prediction = self.logistic.predict(feature_vector)
            if prediction == 0:
                tag = wf.NON_GENE_TAG
            elif prediction == 1:
                tag = wf.GENE_TAG
            new_token = word + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)
