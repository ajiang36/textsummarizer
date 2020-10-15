import nltk
from pprint import pprint

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Tokenizing text, i.e. break up text into lists of words/sentences
text = input ("Enter your text to tokenize here: ")
from nltk.tokenize import word_tokenize, sent_tokenize

sents=sent_tokenize(text)
print(sents)

words=[word_tokenize(sent) for sent in sents]
print(words)

# Removing stopwords, commonly used words like "the", "and", etc.
from nltk.corpus import stopwords
from string import punctuation

customStopWords=set(stopwords.words('english')+list(punctuation))
wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWOStopwords)

# Identifying bigrams, groupings of two consecutive words
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
print(finder.ngram_fd.items())

# Stemming (extract stem of word, e.g. the stem of swimming is swim)
text2 = input ("Enter your text to stem and identify POS here: ")
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)

# POS (part of speech) tagging
tagList = nltk.pos_tag(word_tokenize(text2))
pprint(tagList)

# Disambiguating word meanings
# Words can mean different things depending on their context, and lesk
# uses a word's context to determine which definition of the word
# is being used.
# You can find a description of the algorithm at
# http://www.nltk.org/howto/wsd.html

word = input ("Enter your word here: ")
from nltk.corpus import wordnet as wn

for ss in wn.synsets(word):
   print(ss, ss.definition())

text3 = input ("Enter a sentence with that previous word: ")
from nltk.wsd import lesk
contextualsentence = lesk(word_tokenize(text3), word)
print(contextualsentence, contextualsentence.definition())

text4 = input ("Enter another sentence with that previous word: ")
from nltk.wsd import lesk
contextualsentence = lesk(word_tokenize(text4), word)
print(contextualsentence, contextualsentence.definition())
