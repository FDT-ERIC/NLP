'''
NLTK WordNetLemmatizer（词性还原）
'''

import nltk
nltk.download() #连接外网再尝试打开 NLTK 的 downloader

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

text = ['caresses', 'flies', 'dies', 'meeting', 'stating', 'dating']

singles = [lemmatizer.lemmatize(word) for word in text]

print(singles)