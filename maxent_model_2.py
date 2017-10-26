import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt2:
    def __init__(self, context_word_size):
        self.logistic = LogisticRegression()
        self.cws = context_word_size

    @staticmethod
    def get_feature_vector_for_word(context_words):
        features = [wf.get_feature_vector_for_word(word) for word in context_words]
        feature_vector = np.concatenate(tuple(features), axis=1)
        return feature_vector

    def extract_ngrams_and_labels(self, corpus_path):
        with open(corpus_path) as f:
            ngrams_to_label = {}
            ngrams_list = []
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    tokens = ["_TAG"] * (self.cws - 1) + text.split()
                    for i in xrange(self.cws - 1, len(tokens)):
                        ngram = tuple([tokens[j].split("_")[0] for j in xrange(i - self.cws + 1, i + 1)])
                        ngrams_list.append(ngram)
                        label = tokens[i].split("_")[1]
                        ngrams_to_label[ngram] = label
                else:
                    break
            return ngrams_to_label, ngrams_list

    def extract_data_and_labels(self, corpus_path):
        words_to_label, words_list = MaxEnt2.extract_ngrams_and_labels(corpus_path)
        words_list = [""] * (self.cws - 1) + words_list
        feature_vectors = []
        labels = []
        for i in xrange(self.cws - 1, len(words_list)):
            context_words = [words_list[j] for j in xrange(i - self.cws + 1, i+1)]
            feature_vector = MaxEnt2.get_feature_vector_for_word(context_words)
            feature_vectors.append(feature_vector)
            label = words_to_label[words_list[i]]
            labels.append(label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH * self.cws
        data_matrix = np.zeros((n, d), dtype=bool)
        labels_vector = np.array(labels, dtype=bool)
        for i in range(n):
            data_matrix[i:i + 1, :] = feature_vectors[i]
        return data_matrix, labels_vector

    def train(self, training_data_file):
        train_d, train_l = self.extract_data_and_labels(training_data_file)
        self.logistic.fit(train_d, train_l)

    def tag_text(self, text):
        tokens = text.strip().split(" ")
        new_tokens = []
        words_list = [""]*(self.cws - 1) + [token.split("_")[0] for token in tokens]
        for i in xrange(len(words_list)):
            context_words = [words_list[j] for j in xrange(i - self.cws + 1, i + 1)]
            feature_vector = MaxEnt2.get_feature_vector_for_word(context_words)
            prediction = self.logistic.predict(feature_vector)
            tag = wf.NON_GENE_TAG
            if prediction == 1:
                tag = wf.GENE_TAG
            new_token = words_list[i] + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)
