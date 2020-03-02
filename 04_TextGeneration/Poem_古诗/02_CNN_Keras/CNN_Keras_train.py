import numpy as np
import os
from collections import Counter
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPool1D, Dense
# from conf import *
from warnings import filterwarnings
filterwarnings('ignore')  # 不打印警告


# 语料路径
corpus_path = '../../../Data/04_TextGeneration/Poem_古诗/Keras_Poem.txt'
# 字库大小
len_chr = 1000
# 滑窗大小, 一首诗包括标点符号总共为 24 个字符
window = 24
# 卷积核数量
filters = 25
# 卷积核大小
kernel_size = 5
# 训练总次数
times = 40
# 一次放到内存中的数据量
batch_size = 512
# 完整的数据集通过神经网络的次数
epochs = 25
# 存储模型
modelpath = 'CNN_Keras.h5'


def preprocessing():

    ''' 加载训练语料 '''
    with open(corpus_path, encoding='utf-8') as f:
        # 将古诗词变成一行显示
        seq_chr = f.read().replace('\n', '')

    ''' 数据预处理 '''
    # 语料长度 (372864)
    len_seq = len(seq_chr)
    # 输出的格式: [('，', 31072), ('。', 31072), ('不', 3779), ('人', 3377) ... ]
    chr_ls = Counter(list(seq_chr)).most_common(len_seq)
    # 输出的格式: ['，', '。', '不', '人', ... ]
    chr_ls = [item[0] for item in chr_ls]
    # 字典
    chr_to_id = {c: i for i, c in enumerate(chr_ls)}
    # 字典
    id_to_chr = {i: c for c, i in chr_to_id.items()}
    # 文字序列 --> 索引序列
    seq_id = [chr_to_id[c] for c in seq_chr]
    # yield 迭代返回
    yield len_seq, id_to_chr, seq_id


    ''' 输入层和标签 '''
    reshape = lambda x: np.reshape(x, (-1, window, 1)) / len_chr
    x = [seq_id[i: i+window] for i in range(len_seq-window)]
    # 格式 (372840, 24, 1)
    x = reshape(x)
    y = [seq_id[i+window] for i in range(len_seq-window)]
    # 格式 (372840, 1000)
    y = to_categorical(y, num_classes=len_chr)
    print('x.shape', x.shape, 'y.shape', y.shape)
    # yield 迭代返回
    yield reshape, x, y


(len_seq, id_to_chr, seq_id), (reshape, x, y) = list(preprocessing())


""" 建模 """
if os.path.exists(modelpath):
    print('Loading Model')
    model = load_model(modelpath)
else:
    print('Modeling')
    model = Sequential(name='CNN_Keras')
    model.add(Conv1D(filters, kernel_size*2, padding='same', activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters*2, kernel_size, padding='same', activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(len_chr, activation='softmax'))
    # print(model.summary())  # 打印模型信息
    model.compile('adam', 'categorical_crossentropy')


""" 随机采样 """
""" prep 是个列表，存储了数字，即索引 """
def draw_sample(predictions, temperature):
    pred = predictions.astype('float64')  # 提高精度防报错
    pred = np.log(pred) / temperature
    pred = pred / np.sum(pred)
    # 从多项式分布中提取样本，这里返回相当于 one-hot
    pred = np.random.multinomial(1, pred, 1)
    # 取出最大的值的索引
    return np.argmax(pred)


""" 预测 """
def predict(t, pred=None):
    if pred is None:
        # 在 [0, len_seq-window] 这个区间内随机选一个数
        rand_int = np.random.randint(len_seq-window)
        pred = seq_id[rand_int: rand_int+window]
    if t:
        print('随机采样, 温度: %.1f' % t)
        sample = draw_sample
    else:
        print('贪婪采样')
        sample = np.argmax
    for _ in range(window):
        x_pred = reshape(pred[-window:])  # 窗口滑动
        y_pred = model.predict(x_pred)[0]  # 二维变成一维的
        i = sample(y_pred, t)
        pred.append(i)
    text = ''.join([id_to_chr[i] for i in pred[-window:]])
    print('\033[033m%s\033[0m' % text)


""" 训练及评估 """
for e in range(times):
    print('第%s次' % e)
    model.fit(x, y, batch_size, epochs, verbose=2)
    model.save(modelpath)
    print(str(e+1).center(window*2, '_'))
    # 训练结果展示
    for t in (None, 1, 1.5, 2):
        predict(t)
