import pandas as pd

# 读取和处理数据
data = pd.read_csv("../Data/02_NER_Greedy/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill") # 向下填充空缺数据


# 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin

# Majority Voting，重写分类器
class MajorityVotingTagger(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """
        X: list of words
        y: list of tags
        """
        # 遍历数据，计算 word 属于什么 tag，并计数，用于之后的多数决
        word2cnt = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in word2cnt:
                if t in word2cnt[x]:
                    word2cnt[x][t] += 1
                else:
                    word2cnt[x][t] = 1
            else:
                word2cnt[x] = {t: 1}

        # 构建字典，多数决，用于决定这个 word 属于什么 tag
        self.mjvote = {}
        for k, d in word2cnt.items():
            self.mjvote[k] = max(d, key=d.get)

    def predict(self, X, y=None):
        '''
        Predict the the tag from memory. If word is unknown, predict 'O'.
        '''
        return [self.mjvote.get(x, 'O') for x in X]


# 将数据处理成需要的格式
words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()

# 预测，输出准确率
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
pred = cross_val_predict(estimator=MajorityVotingTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report)