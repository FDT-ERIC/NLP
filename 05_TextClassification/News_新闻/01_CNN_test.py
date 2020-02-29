from tensorflow.python.keras.models import load_model
import re, jieba
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle


PATH = '../../Data/05_TextClassification/News_新闻/01_W2V_CNN_GRU/'

# 用户输入文本
st = str(input('用户输入: '))

# 对文本进行过滤
st = st.strip().lower()
st = re.sub('\d+[.:年月日时分秒]+', '', st)
st = re.sub('[^\u4e00-\u9fa5a-zA-Z]+', '', st)

# 过滤停用词
with open(PATH + 'StopWords/stopwords.txt', encoding='utf-8') as f:  # 中文停词
    stopwords = set(f.read().split())
    stopwords |= set(ENGLISH_STOP_WORDS)

words = [w for w in jieba.cut(st) if w not in stopwords]

# 最终文本形式
ls_of_words = []
ls_of_words.append(words)

# 加载训练好的 词向量 模型
word2vec = Word2Vec.load(PATH + 'W2V_Model/W2V_Model.model')
# 获取总的词向量
vectors = word2vec.wv.vectors

# 通过提供的 index2word() 方法，构建 word2index 字典
word2index = {w: i for i, w in enumerate(word2vec.wv.index2word)}

# 将 词 转变成 对应的 词向量
X = [[word2index[w] for w in x if w in word2index] for x in ls_of_words]
X = [x for x in X if x]
X = [[vectors[i] for i in x] for x in X]
# pad_sequences 对应于模型输入要求
X = pad_sequences(X, 200, dtype='float')

# 加载训练好的 CNN 模型
cnn_model = load_model(PATH + 'W2V_CNN_Model/W2V_CNN.h5')

# 预测的结果 (输出类别的 [index])
label_index = cnn_model.predict_classes(X)


# 读取 id2label，从而拿出 index 对应的 label
def id2label():
    # 以二进制文本形式读取
    with open(PATH + 'Id_to_Label/id2label', 'rb') as f:
        # load() 反序列化对象，将文件中的数据解析为一个 Python 对象
        return pickle.load(f)
id_to_label = id2label()

label = id_to_label[label_index[0]].replace('_', '')

print('所属类别为: ', label)
