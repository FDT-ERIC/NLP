import pandas as pd
import numpy as np

# 读取和处理数据
data = pd.read_csv("../Data/02_NER_Greedy/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill") # 向下填充空缺数据

# 提取特征，作为随机森林的特征选择
'''
istitle() 方法检测字符串中所有的单词拼写首字母是否为大写，且其他字母为小写
islower() 方法检测字符串是否由小写字母组成
isupper() 方法检测字符串是否由大写字母组成
isdigit() 方法检测字符串是否只由数字组成
isalpha() 方法检测字符串是否只由字母组成
'''
def get_feature(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(),  word.isalpha()])

# 将数据处理成需要的格式
words = [get_feature(w) for w in data["Word"].values.tolist()]
tags = data["Tag"].values.tolist()

# 预测，输出准确率
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
# 参数 n_estimators 表示森林中树的数量
pred = cross_val_predict(RandomForestClassifier(n_estimators=20), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report)