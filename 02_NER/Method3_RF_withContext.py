import pandas as pd
import numpy as np
from Method1_Majority_Voting import MajorityVotingTagger
from sklearn.preprocessing import LabelEncoder

out = []
y = []
mv_tagger = MajorityVotingTagger()
tag_encoder = LabelEncoder()
pos_encoder = LabelEncoder()

# 读取和处理数据
data = pd.read_csv("../Data/02_NER_Greedy/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill") # 向下填充空缺数据
words = data["Word"].values.tolist()
pos = data["POS"].values.tolist()
tags = data["Tag"].values.tolist()

# 先 fit（拟合）一下
mv_tagger.fit(words, tags)
tag_encoder.fit(tags)
pos_encoder.fit(pos)

# 为了使用上下文，将单词组合成句子
def get_sentences(data):
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
    sentence_grouped = data.groupby("Sentence #").apply(agg_func)
    return [s for s in sentence_grouped]

sentences = get_sentences(data)


# 使用上一个单词和下一个单词的信息，作为 RF 的特征，相当于考虑了时序的信息
for sentence in sentences:
    for i in range(len(sentence)):
        w, p, t = sentence[i][0], sentence[i][1], sentence[i][2]

        if i < len(sentence) - 1:
            # 如果不是最后一个单词，则可以用到下文的信息
            mem_tag_r = tag_encoder.transform(mv_tagger.predict([sentence[i + 1][0]]))[0]
            true_pos_r = pos_encoder.transform([sentence[i + 1][1]])[0]
        else:
            mem_tag_r = tag_encoder.transform(['O'])[0]
            true_pos_r = pos_encoder.transform(['.'])[0]

        if i > 0:
            # 如果不是第一个单词，则可以用到上文的信息
            mem_tag_l = tag_encoder.transform(mv_tagger.predict([sentence[i - 1][0]]))[0]
            true_pos_l = pos_encoder.transform([sentence[i - 1][1]])[0]
        else:
            mem_tag_l = tag_encoder.transform(['O'])[0]
            true_pos_l = pos_encoder.transform(['.'])[0]
        # print (mem_tag_r, true_pos_r, mem_tag_l, true_pos_l)

        out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                             tag_encoder.transform(mv_tagger.predict([sentence[i][0]])),
                             pos_encoder.transform([p])[0], mem_tag_r, true_pos_r, mem_tag_l, true_pos_l]))
        y.append(t)

# 预测，准确率
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
pred = cross_val_predict(RandomForestClassifier(n_estimators=20), X=out, y=y, cv=5)
report = classification_report(y_pred=pred, y_true=y)
print(report)