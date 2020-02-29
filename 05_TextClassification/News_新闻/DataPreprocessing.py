import os, pickle

# 总路径
PATH = '../../Data/05_TextClassification/News_新闻/01_W2V_CNN_GRU/'

# 训练语料路径
PATH_CORPUS = PATH + 'Corpus/'

# 存放 label
PATH2 = PATH + 'Id_to_Label/'
PATH_LABEL = PATH2 + 'id2label'

# 存放 data 和 vector
PATH3 = PATH + 'Xy_Vector/'
PATH_XY_VEC = PATH3 + 'Xy_vec'
PATH_XY_VEC_TFIDF = PATH3 + 'Xy_vec_plus_TfIdf'

# 停用词
PATH_StopWord = PATH + 'StopWords/stopwords.txt'


# 词向量参数
size = 100  # 维度
window = 10  # 划窗大小
min_count = 1  # 过滤低频词


# 取出数据的 标签 并存储，形式为 {index: label}
def id2label():
    if not os.path.exists(PATH_LABEL):
        labels = {i: l.replace('.txt', '_') for i, l in enumerate(os.listdir(PATH_CORPUS))}
        # 以二进制文本形式写入
        with open(PATH_LABEL, 'wb') as f:
            # dump() 序列化对象，并将结果数据流写入到文件对象中
            pickle.dump(labels, f)
    # 以二进制文本形式读取
    with open(PATH_LABEL, 'rb') as f:
        # load() 反序列化对象，将文件中的数据解析为一个 Python 对象
        return pickle.load(f)


# word2vec + Tfidf  (文件很大)
# def load_xy(i=0):
#     path = [PATH_XY_VEC, PATH_XY_VEC_TFIDF]
#     if not os.path.exists(PATH_XY_VEC_TFIDF):
#         word2vector()
#     with open(path[i], 'rb') as f:
#         return pickle.load(f)


# word2vec
def load_xy(i=0):
    path = [PATH_XY_VEC]
    if not os.path.exists(PATH_XY_VEC):
        word2vector()
    with open(path[i], 'rb') as f:
        return pickle.load(f)


# 处理数据，并将数据集分割，测试集的比例是 20%
def text_filter():
    import re, jieba
    from sklearn.model_selection import train_test_split  # 分割数据集
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS  # 英文停词
    with open(PATH_StopWord, encoding='utf-8') as f:  # 中文停词
        stopwords = set(f.read().split())
        stopwords |= set(ENGLISH_STOP_WORDS)
    # {index: label}
    labels = id2label()
    ls_of_words = []
    ls_of_label = []
    for label_id, label in labels.items():
        print(label_id, label)
        with open(PATH_CORPUS + label.replace('_', '.txt'), encoding='utf-8') as f:
            for line in f.readlines():
                # 获取该行的有效字符
                line = ''.join(re.findall('^\[(.+)\]$', line)).strip().lower()
                # 把 时间 替换成 空格
                line = re.sub('\d+[.:年月日时分秒]+', ' ', line)
                # 只能存在 汉字 和 英文字母
                line = re.sub('[^\u4e00-\u9fa5a-zA-Z]+', ' ', line)
                # 分词
                words = [w for w in jieba.cut(line) if w.strip() and w not in stopwords]
                if words:
                    ls_of_words.append(words)
                    ls_of_label.append(label_id)
    # 分割数据集，测试集的比例是 20%
    # 格式为: train_data, test_data, train_label, test_label
    # data 是 二维数组，label 是 一维数组
    return train_test_split(ls_of_words, ls_of_label, test_size=0.2)


# 词向量训练和存储
def word2vector():
    from gensim.models import Word2Vec, TfidfModel
    # from collections import Counter
    # import numpy as np

    # X 代表 数据集，y 代表 标签
    X_train, X_test, y_train, y_test = text_filter()


    """词向量"""
    word2vec = Word2Vec(sentences=X_train, size=size, window=window, min_count=min_count)
    word2vec.save('W2V_Model.model')  # 保存模型

    # 通过提供的 index2word() 方法，构建 word2index 字典
    word2index = {w: i for i, w in enumerate(word2vec.wv.index2word)}

    # 词向量
    vectors = word2vec.wv.vectors

    # 将 词 转变成 对应的 index索引
    X_train = [[word2index[w] for w in x if w in word2index] for x in X_train]
    X_test = [[word2index[w] for w in x if w in word2index] for x in X_test]

    # 过滤空值
    X_train = [x for x in X_train if x]
    X_test = [x for x in X_test if x]
    y_train = [y for y, x in zip(y_train, X_train) if x]
    y_test = [y for y, x in zip(y_test, X_test) if x]

    # 保存 Word2Vec 词向量
    with open(PATH_XY_VEC, 'wb') as f:
        # dump() 序列化对象，并将结果数据流写入到文件对象中
        pickle.dump((
            [[vectors[i] for i in x] for x in X_train],
            [[vectors[i] for i in x] for x in X_test],
            y_train, y_test
        ), f)

    """词向量 + TfIdf"""
    # TfIdf = TfidfModel([Counter(x).most_common() for x in X_train])
    # Idfs = np.array([[TfIdf.idfs[i]] for i in range(len(word2index))])
    # vectors = vectors * Idfs
    #
    # 保存 Word2Vec 词向量 + TfIdf
    # with open(PATH_XY_VEC_TFIDF, 'wb') as f:
    #     pickle.dump((
    #         [[vectors[i] for i in x] for x in X_train],
    #         [[vectors[i] for i in x] for x in X_test],
    #         y_train, y_test
    #     ), f)


if __name__ == '__main__':
    word2vector()

