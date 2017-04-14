# coding: utf-8
import pymystem3
import pymorphy2
import wikipedia
import nltk
import re

text = open('wiki.txt','r').read()

STOP_WORDS_SET = set()
PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
CARRIAGE_RETURNS = ['\n', '\r\n']
WORD_REGEX = u"^[а-ясё]+$"

stopwordsfile = open("stopwords_rus.txt", "r")
for word in stopwordsfile:
    word = word.replace("\n", '')
    word = word.replace("\r\n", '')
    STOP_WORDS_SET.add(word)

tokens=[]
for word in text.split(' '):
    word = word.decode('utf-8').lower()
    word = word.encode('utf-8')
    for punc in PUNCTUATION + CARRIAGE_RETURNS:
        word = word.replace(punc, '').strip("'")
    if re.match(WORD_REGEX, word.decode('utf-8')):
        if word and (word not in STOP_WORDS_SET) and (len(word) > 1):
            tokens.append(word)

morph = pymorphy2.MorphAnalyzer()
noun = []
for token in tokens:
    POS = str(morph.parse(token.decode('utf-8'))[0].tag.POS)
    if POS == 'NOUN':
        noun.append(token)

fd = nltk.FreqDist(noun)
for n in fd:
    print (n)




