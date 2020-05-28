import csv
import copy
from os import listdir
from os.path import isfile, join

from text_preprocessing import *

wordsim_vocabulary = set()

DIRECTORY_NAME = 'data/wordsim_corpus/'

with open('data/wordsim_similarity_goldstandard.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        wordsim_vocabulary.add(row[0].lower())
        wordsim_vocabulary.add(row[1].lower())

print('Vocabulary size: ', len(wordsim_vocabulary))
wordsim_list = list(wordsim_vocabulary)
wordsim_list.sort()
print('Vocabulary')
print(wordsim_list)


words_not_presented_in_documents = copy.copy(wordsim_vocabulary)

document_names = [f for f in listdir(DIRECTORY_NAME) if isfile(join(DIRECTORY_NAME, f))]
for name in document_names:
    print("Process document ", name) 
    with open(join(DIRECTORY_NAME, name), 'r') as f:
        document = f.read()
        sentences = tokenize_text(document)
        for sentence in sentences:
            tokens = lemmatize_text(sentence)
            for token in tokens:
                words_not_presented_in_documents.discard(token)
    print('After document left {} words'.format(len(words_not_presented_in_documents)))

left_list = list(words_not_presented_in_documents)
left_list .sort()
print("Left to find")
print(left_list)

print('Vocabulary size: {0}. Find in documents: {1}. Left to find: {2}.'.format(len(wordsim_vocabulary), len(wordsim_vocabulary) - len(words_not_presented_in_documents), len(words_not_presented_in_documents)))




