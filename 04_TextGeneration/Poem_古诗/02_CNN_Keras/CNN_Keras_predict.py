import numpy as np
from collections import Counter
from keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')  # 不打印警告


# 语料路径
corpus_path = '../../../Data/04_TextGeneration/Poem_古诗/Keras_Poem.txt'
# 字库大小
len_chr = 1000
# 滑窗大小, 一首诗包括标点符号总共为 24 个字符
window = 24
# 存储模型
modelpath = 'CNN_Keras.h5'


# 数据预处理
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
    # 获取 字母c 对应的 index值，若没有则随机附一个值
    c2i = lambda c: chr_to_id.get(c, np.random.randint(len_chr))
    # 文字序列 --> 索引序列
    seq_id = [chr_to_id[c] for c in seq_chr]
    # 将数据 reshape 成输入的格式
    reshape = lambda x: np.reshape(x, (-1, window, 1)) / len_chr
    # 返回
    return len_seq, id_to_chr, seq_id, c2i, reshape


# 语料加载和处理
len_seq, id_to_chr, seq_id, c2i, reshape = preprocessing()
# 模型加载
model = load_model(modelpath)


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
def predict(t, pred):
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


# 输入输出
if __name__ == '__main__':
    for t in (None, 1, 1.5, 2):
        predict(t, seq_id[-window:])
    while True:
        title = input('输入标题: ').strip() + '。'
        len_t = len(title)
        randint = np.random.randint(len_seq-window + len_t)
        randint = int(randint // 12 * 12)
        pred = seq_id[randint: randint+window-len_t] + [c2i(c) for c in title]
        for t in (None, 1, 1.5, 2):
            predict(t, pred)
