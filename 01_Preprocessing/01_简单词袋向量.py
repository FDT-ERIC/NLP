# 方法1: 词袋模型 (按照词语出现的个数)
'''
输出:
[[0 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 0 1 0 1 0]
 [1 0 0 1 0 1 0 0 2 0 0 1 0 0 1 0 1 0 0 0 0]
 [0 1 0 0 1 0 0 0 0 1 0 0 1 1 0 2 0 0 2 0 1]]

['actually', 'and', 'beijing', 'but', 'car', 'denied', 'from', 'going', 'he', 'in', 'is', 'lied', 'lost', 'mike', 'my', 'phone', 'request', 'shanghai', 'the', 'to', 'was']
'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = ['He is going from Beijing to Shanghai.',
          'He denied my request, but he actually lied.',
          'Mike lost the phone, and phone was in the car.']
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())
print(X)


# 方法2: 词袋模型 (tf-idf)
'''
tf(文档 d 中 w 的词频)
idf(log(N:语料库中的文档总数 / N(w):词语 w 出现在多少个文档中 ))
'''
'''
输出:
[[0.         0.         0.39379499 0.         0.         0.
  0.39379499 0.39379499 0.26372909 0.         0.39379499 0.
  0.         0.         0.         0.         0.         0.39379499
  0.         0.39379499 0.        ]
 [0.35819397 0.         0.         0.35819397 0.         0.35819397
  0.         0.         0.47977335 0.         0.         0.35819397
  0.         0.         0.35819397 0.         0.35819397 0.
  0.         0.         0.        ]
 [0.         0.26726124 0.         0.         0.26726124 0.
  0.         0.         0.         0.26726124 0.         0.
  0.26726124 0.26726124 0.         0.53452248 0.         0.
  0.53452248 0.         0.26726124]]
  
['actually', 'and', 'beijing', 'but', 'car', 'denied', 'from', 'going', 'he', 'in', 'is', 'lied', 'lost', 'mike', 'my', 'phone', 'request', 'shanghai', 'the', 'to', 'was']

'''
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(smooth_idf=False)
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())