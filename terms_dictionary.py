import pygtrie
from heapq import heapify, heappush, heappop
from os import listdir
from os.path import isfile, join

from text_preprocessing import *


class DictionaryTrie:
    def __init__(self):
        self.trie = pygtrie.CharTrie()
        self.size = 0

    def find_n_max_elements(self, n):
        result_heap = []
        heapify(result_heap)
        for key, value in self.trie.iteritems():
            if len(result_heap) < n:
                heappush(result_heap, (value, key))
            else:
                head = heappop(result_heap)
                if value >= head[0]:
                    heappush(result_heap, (value, key))
                else:
                    heappush(result_heap, head)
        return result_heap

    def get_items_more_than_threshold(self, threshold):
        items = []
        for key, value in self.trie.iteritems():
            if value > threshold:
                items.append((key, value))
        return items

    def get_items_less_than_threshold(self, threshold):
        items = []
        for key, value in self.trie.iteritems():
            if value < threshold:
                items.append((key, value))
        return items

    def extend(self, another_trie):
        for key,value in another_trie.iteritems():
            self.trie[key] = self.trie.get(key, 0) + value
            self.size += value

    def __getitem__(self, item):
        return self.trie[item]

    def __setitem__(self, key, value):
        self.trie[key] = value

    def iteritems(self):
        return self.trie.iteritems()

    def add_words(self, words):
        for w in words:
            self.trie[w] = self.trie.get(w, 0) + 1
        self.size += len(words)

    def get(self, key, default=None):
        return self.trie.get(key, default)

    def __delitem__(self, key_or_slice):
        del self.trie[key_or_slice]



    @staticmethod
    def build_document_dictionary(doc_path):
        corpus = DictionaryTrie()
        with open(doc_path, 'r') as doc:
            text = doc.read()
            tokens = tokenize_text(text)
            lemmas = lemmatize_text(tokens)
            corpus.add_words(lemmas)
        return corpus

    @staticmethod
    def build_category_dictionary(category_path):
        document_list = [f for f in listdir(category_path) if isfile(join(category_path, f))]
        corpus = DictionaryTrie()
        for document in document_list:
            with open(join(category_path, document), 'r') as doc:
                text = doc.read()
                tokens = tokenize_text(text)
                lemmas = lemmatize_text(tokens)
                corpus.add_words(lemmas)
        return corpus
