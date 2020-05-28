import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def tokenize_text(text):
    result = []
    for sentence in sent_tokenize(text):
        sentence = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = word_tokenize(sentence.lower())
        result.append(tokens)
    return result

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = list()
    for token, pos in pos_tag(tokens):
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos=='':
            lemmas.append(token)
        else:
            lemmas.append(lemmatizer.lemmatize(token, get_wordnet_pos(pos)))
    return lemmas

def remove_stop_words(tokens):
    stopwords_set = set(stopwords.words('english'))
    return [token for token in tokens if token not in stopwords_set]
