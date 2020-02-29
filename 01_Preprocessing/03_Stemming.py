'''
NLTK PorterStemmer（词性还原，有个缺点，通过 PorterStemmer 处理的词不一定存在词库里）
'''

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

text = ['caresses', 'flies', 'dies', 'meeting', 'stating', 'dating']

singles = [stemmer.stem(word) for word in text]

print(' '.join(singles))