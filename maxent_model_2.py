import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt2:
    def __init__(self, context_word_size):
        self.logistic = LogisticRegression()
        self.cws = context_word_size

    @staticmethod
    def add_new_and_remove_oldest(array, new_value):
        array[:0] = [new_value]
        array.pop()
        return array

    def get_feature_vector_for_ngram(self, ngram, prev_labels):
        vector = np.zeros((1, wf.FEATURE_VECTOR_SIZE * self.cws + (self.cws - 1)))
        for i in xrange(len(ngram)):
            word = ngram[i]
            word_index_offset = i * wf.FEATURE_VECTOR_SIZE
            for j, c in enumerate(word):
                if j >= wf.MAXIMUM_WORD_LENGTH:
                    break
                char_index_offset = j * wf.NUM_ACCEPTED_CHARACTERS
                if c in wf.char_index_map.keys():
                    char_value_offset = wf.char_index_map[c]
                    feature_index = word_index_offset + char_index_offset + char_value_offset
                    vector[:, feature_index] = 1
        for k, label in enumerate(prev_labels):
            label_offset = self.cws * wf.FEATURE_VECTOR_SIZE + k
            vector[:, label_offset:label_offset+1] = label
        return vector

    def extract_ngrams_and_labels(self, corpus_path):
        with open(corpus_path) as f:
            ngrams_to_label = {}
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    tokens = ["_TAG"] * (self.cws - 1) + text.split(" ")
                    for i in xrange(self.cws - 1, len(tokens)):
                        ngram = tuple([tokens[j].split("_")[0] for j in xrange(i - self.cws + 1, i + 1)])
                        tag = tokens[i].split("_")[1]
                        ngrams_to_label[ngram] = wf.get_label_for_tag(tag)
                else:
                    break
            return ngrams_to_label

    def extract_data_and_labels(self, corpus_path):
        ngrams_to_labels = self.extract_ngrams_and_labels(corpus_path)
        feature_vectors = []
        labels = []
        prev_labels = [2]*(self.cws-1)
        for ngram, label in ngrams_to_labels.iteritems():
            feature_vector = self.get_feature_vector_for_ngram(ngram, prev_labels)
            feature_vectors.append(feature_vector)
            label = ngrams_to_labels[ngram]
            labels.append(label)
            prev_labels = MaxEnt2.add_new_and_remove_oldest(prev_labels, label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH * self.cws + (self.cws - 1)
        data_matrix = np.zeros((n, d))
        labels_vector = np.array(labels)
        for i in range(n):
            data_matrix[i:i + 1, :] = feature_vectors[i]
        return data_matrix, labels_vector

    def train(self, training_data_file):
        train_d, train_l = self.extract_data_and_labels(training_data_file)
        self.logistic.fit(train_d, train_l)

    def greedy_tag_text(self, text):
        tokens = text.strip().split(" ")
        new_tokens = []
        words_list = [""] * (self.cws - 1) + [token.split("_")[0] for token in tokens]
        prev_labels = [2] * (self.cws - 1)
        for i in xrange(len(words_list)):
            context_words = tuple([words_list[j] for j in xrange(i - self.cws + 1, i + 1)])
            feature_vector = self.get_feature_vector_for_ngram(context_words, prev_labels)
            prediction = self.logistic.predict(feature_vector)
            tag = wf.NON_GENE_TAG
            if prediction == 1:
                tag = wf.GENE_TAG
            new_token = words_list[i] + "_" + tag
            new_tokens.append(new_token)
            prev_labels = MaxEnt2.add_new_and_remove_oldest(prev_labels, prediction)
        return " ".join(new_tokens)

    def viterbi_tag_text(self, text):
        #todo: figure out how to pass prev_label values ot self.get_feature_vector...(context_words, prev_labels)
        is_gene_table = [1]
        not_gene_table = [1]
        is_gene_labels = []
        not_gene_lables = []
        prev_labels = [2] * (self.cws - 1)
        tokens = text.strip().split(" ")
        words_list = [""] * (self.cws - 1) + [token.split("_")[0] for token in tokens]
        for i in xrange(len(words_list)):
            context_words = tuple([words_list[j] for j in xrange(i - self.cws + 1, i + 1)])
            prev_labels[-1] = 1
            prev_word_is_gene_fv = self.get_feature_vector_for_ngram(context_words, prev_labels)
            prev_word_is_gene_distribution = self.logistic.predict_proba(prev_word_is_gene_fv)[0]

            prev_labels[-1] = 0
            prev_word_not_gene_fv = self.get_feature_vector_for_ngram(context_words, prev_labels)
            prev_word_not_gene_distribution = self.logistic.predict_proba(prev_word_not_gene_fv)[0]

            prob_is_gene_and_prev_is_gene = prev_word_is_gene_distribution[1] * is_gene_table[-1]
            prob_is_gene_and_prev_not_gene = prev_word_not_gene_distribution[1] * not_gene_table[1]
            if prob_is_gene_and_prev_is_gene > prob_is_gene_and_prev_not_gene:
                is_gene_table.append(prob_is_gene_and_prev_is_gene)
                is_gene_labels.append(1)
            else:
                is_gene_table.append(prob_is_gene_and_prev_not_gene)
                is_gene_labels.append(0)


            prob_not_gene_and_prev_is_gene = prev_word_is_gene_distribution[0] * is_gene_table[-1]
            prob_not_gene_and_prev_not_gene = prev_word_not_gene_distribution[0] * not_gene_table[1]
            if prob_not_gene_and_prev_is_gene > prob_not_gene_and_prev_not_gene:
                not_gene_table.append(prob_not_gene_and_prev_is_gene)
                not_gene_lables.append(1)
            else:
                not_gene_table.append(prob_not_gene_and_prev_not_gene)
                not_gene_lables.append(0)


            prev_labels = MaxEnt2.add_new_and_remove_oldest(prev_labels, pass)

    def tag_text(self, text, viterbi=False):
        return self.viterbi_tag_text(text) if viterbi else self.greedy_tag_text(text)

