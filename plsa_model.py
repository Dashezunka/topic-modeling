import sys
from os.path import join, isfile
from os import listdir
import csv
import numpy as np

from plsa_options import *
from text_preprocessing import *
from terms_dictionary import DictionaryTrie
from similarity_metrics import *

# Build document corpus, consisting of tuples (doc_name, doc_sentences_list)
def build_corpus():
    corpus = []
    document_name_list = [f for f in listdir(CORPUS_PATH) if isfile(join(CORPUS_PATH, f))]
    for name in document_name_list:
        with open(join(CORPUS_PATH, name), 'r') as doc:
            text = doc.read()
            sentences = tokenize_text(text)
            document_sentences = []
            for sentence in sentences:
                lemmas = lemmatize_text(sentence)
                preprocessed_sentence = remove_stop_words(lemmas)
                if len(preprocessed_sentence) < 2:
                    continue
                document_sentences.extend(preprocessed_sentence)
            corpus.append((name, document_sentences))
    # Form corpus vocabulary
    corpus_vocabulary = set()
    for document_name, document in corpus:
        corpus_vocabulary.update(document)
    corpus_vocabulary = list(corpus_vocabulary)
    return corpus, corpus_vocabulary

# Build frequency dicts for documents
def build_frequency_dict(corpus):
    document_frequency_dicts = {}
    for document_name, document in corpus:
        frequency_dict = DictionaryTrie()
        frequency_dict.add_words(document)
        document_frequency_dicts[document_name] = frequency_dict
    return document_frequency_dicts

def normalize_vector(vec):
    sum_of_elements = np.sum(vec)
    if sum_of_elements == 0:
        print('Divide to zero!')
        sys.exit(1)
    return np.array([elem / sum_of_elements for elem in vec])

# Compare model results for wordsim words with gold standard values
def compare_with_wordsim(topic_word_matrix, vocab):
    with open('data/wordsim_similarity_goldstandard.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        result = []
        for row in csv_reader:
            word1 = row[0].lower()
            word2 = row[1].lower()
            wordsim_similarity = row[2]
            # check if both words from wordsim exist in model dictionary
            model_similarity = None

            if word1 in vocab and word2 in vocab:
                word1_index = vocab.index(word1)
                word2_index = vocab.index(word2)
                w1_vec = topic_word_matrix[:, word1_index]
                w2_vec = topic_word_matrix[:, word2_index]
                model_similarity = calc_cosine_similarity(w1_vec, w2_vec)
            result.append((word1, word2, wordsim_similarity, model_similarity))
        return result

def plsa(corpus, vocabulary, document_frequency_dicts, number_of_topics, max_iterations=5):
    nummber_of_documents = len(corpus)
    vocabulary_size = len(vocabulary)
    # Matrices initialization
    doc_topic_prob_matrix = np.zeros([nummber_of_documents, number_of_topics], dtype=np.float)
    topic_word_prob_matrix = np.zeros([number_of_topics, vocabulary_size], dtype=np.float)
    topic_probability_matrix = np.zeros([nummber_of_documents, vocabulary_size, number_of_topics], dtype=np.float)

    # Fill matrices with random values
    doc_topic_prob_matrix = np.random.random(size=(nummber_of_documents, number_of_topics))
    topic_word_prob_matrix = np.random.random(size=(number_of_topics, vocabulary_size))

    # Normalize matrices
    doc_topic_prob_matrix = normalize_vector(doc_topic_prob_matrix)
    topic_word_prob_matrix = normalize_vector(topic_word_prob_matrix)

    # pLSA iterations
    for iteration_index in range(max_iterations):
        print("EM: Iteration {0}".format(iteration_index))
        # Expectation step
        for doc_index, (document_name, document_sentences) in enumerate(corpus):
            for word_index in range(vocabulary_size):
                doc_words_probs = doc_topic_prob_matrix[doc_index, :] * topic_word_prob_matrix[:, word_index]
                topic_probability_matrix[doc_index, word_index] = normalize_vector(doc_words_probs)
        # Maximization for [D x T]
        for doc_index, (document_name, document_sentences) in enumerate(corpus):
            for topic_index in range(number_of_topics):
                s = 0
                for word_index, word in enumerate(vocabulary):
                    frequency = document_frequency_dicts[document_name].get(word, 0)
                    s += frequency * topic_probability_matrix[doc_index, word_index, topic_index]
                doc_topic_prob_matrix[doc_index, topic_index] = s
            doc_topic_prob_matrix[doc_index] = normalize_vector(doc_topic_prob_matrix[doc_index])

        # Maximization for [T x W]
        for topic_index in range(number_of_topics):
            for word_index, word in enumerate(vocabulary):
                s = 0
                for doc_index, (document_name, document_sentences) in enumerate(corpus):
                    frequency = document_frequency_dicts[document_name].get(word, 0)
                    s += frequency * topic_probability_matrix[doc_index, word_index, topic_index]
                topic_word_prob_matrix[topic_index, word_index] = s
            topic_word_prob_matrix[topic_index] = normalize_vector(topic_word_prob_matrix[topic_index])

    return doc_topic_prob_matrix, topic_word_prob_matrix

# Read documents and build corpus of documents
corpus, corpus_vocabulary = build_corpus()
print("Corpus has {0} words".format(sum([len(doc) for doc_name, doc in corpus])))
print("Corpus has unique words {0}".format(len(corpus_vocabulary)))

# Build frequency dicts dor docs and print top frequency words for every doc
document_frequency_dicts = build_frequency_dict(corpus)
for doc_index, (document_name, document) in enumerate(document_frequency_dicts.items()):
    print("{0}. For document '{1}'. Top frequency words are {2}.".format(doc_index, document_name,
                                                                         [word for freq, word in
                                                                          document.find_n_max_elements(5)]))
# pLSA algorithm
doc_top_matrix, topic_word_matrix = plsa(corpus, corpus_vocabulary,  document_frequency_dicts, TOPICS_NUMBER, EM_ITERATIONS)

# Print matrices
for index, row in enumerate(doc_top_matrix):
    print(list(document_frequency_dicts.keys())[index], [round(element, 2) for element in row])

for index, row in enumerate(topic_word_matrix):
    word_probabilities = [(word_probability, corpus_vocabulary[word_index]) for word_index, word_probability in enumerate(row)]
    word_probabilities.sort(reverse=True)
    print("Topic ", index)
    print([(round(probability, 2), word) for probability, word in word_probabilities[:15]])
print('The end!')

# Save matrices as csv
print("Save matrices as csv")
with open('data/results/doc_topics_distrib.csv', 'w+') as out:
    doc_top_matrix_out = csv.writer(out)
    topic_str = 'topic {0} prob'
    topic_probs_header = [topic_str.format(i) for i in range(TOPICS_NUMBER)]
    doc_top_matrix_out.writerow(['document name'] + topic_probs_header)
    for index, row in enumerate(doc_top_matrix):
        doc_top_matrix_out.writerow([list(document_frequency_dicts.keys())[index]] + [round(element, 2) for element in row])

print("Save matrices as csv")
with open('data/results/topic_word_distrib.csv', 'w+') as out:
    top_word_matrix_out = csv.writer(out)
    topic_str = 'top {0} word'
    topic_probs_header = [topic_str.format(i) for i in range(TOP_PROB_WORD)]
    top_word_matrix_out.writerow(['topic'] + topic_probs_header)
    for index, row in enumerate(topic_word_matrix):
        word_probabilities = [(word_probability, corpus_vocabulary[word_index]) for word_index, word_probability in
                              enumerate(row)]
        word_probabilities.sort(reverse=True)

        top_word_matrix_out.writerow(['topic {0}'.format(index) ] +
                                    [(round(probability, 2), word) for probability, word in word_probabilities[:TOP_PROB_WORD]])

# Make a comparison of model results and gold standard values for words from wordsim353 corpus.
compare_with_wordsim(topic_word_matrix, corpus_vocabulary)