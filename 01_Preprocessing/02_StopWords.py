'''
NLTK StopWords（停用词）
'''

import nltk

stopwords = nltk.corpus.stopwords.words("english")

print(stopwords[:50])