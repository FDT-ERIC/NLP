import numpy as np
import torch
import random
from collections import Counter


# 数据预处理
def preprocess(text, freq):
    '''
    :param text: 输入的文本
    :param freq: 词出现的最少次数，用于过滤低频词
    :return:
    '''
    # 开始对数据进行清洗，各种骚操作预处理
    text = text.lower()
    text = text.replaceD(".", "<PERIO>")
    ...
    words = text.split() # 分词，一般用 jieba 来分

    # 计算词出现的次数，返回的是字典形式
    word_counts = Counter(words)
    # 过滤低频词
    trimmed_words = [word for word in words if word_counts[word]>freq]

    return trimmed_words


# 加载训练语料
with open('file_path') as f:
    text = f.read()

# 文本预处理
words = preprocess(text)
# 去重
vocab = set(words)
# 词 到 index 的映射
word_to_idx = {w: c for c, w in enumerate(vocab)}
# index 到 词 的映射
idx_to_vocab = {c: w for c, w in enumerate(vocab)}
# 所有词的索引
idx_words = [word_to_idx[w] for w in words]


t = 1e-5
# 计算所有 index 出现的次数，返回的是字典形式
idx_word_counts = Counter(idx_words)
# 训练语料的长度
total_count = len(idx_words)
# 计算词频，格式为 {'词的index': '在所有的词中出现的比例'}
word_freqs = {w: c/total_count for w, c in idx_word_counts.items()}

# 负采样的单词分布，根据 Word2Vec 的作者写的论文来的
# noise_dist 形式如 [0.2, 0.3, 0.5], 表示对应位置的概率
word_freqs = np.array(word_freqs.values())
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75) / np.sum(unigram_dist**(0.75)))

# 每个词去除的概率
prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in idx_word_counts}


# 最后给出训练的词
train_words = [w for w in idx_words if random.random()<(1-prob_drop[w])]



# 获取 中心词(center) 的 周边词(context)
def get_target(words, idx, window_size=5):
    '''
    :param words: 一个 batch 里的 词
    :param idx: 中心词对应的下标
    :param window_size: 滑窗大小
    :return:
    '''
    # 随机选取 <= window_size 的窗口值，如这里可以是 1,2,3,4,5
    target_window = np.random.randint(1, window_size+1)
    # 窗口的开始下标
    start_point = idx-target_window if (idx-target_window)>0 else 0
    # 窗口的结束下标
    end_point = idx+target_window
    # 去重
    targets = set(words[start_point:idx]+words[idx+1:end_point+1])
    return list(targets)


# 构建迭代器
def get_batch(words, batch_size, window_size):
    '''
    :param words: 训练数据
    :param batch_size: 一次性放到内存占的数据量
    :param window_size: 滑窗大小
    :return:
    '''
    # 总共有多少个 batch
    n_batches = len(words)//batch_size
    # 需要取整，填不满最后的 batch 的那些 词 就不要了
    words = words[:n_batches*batch_size]
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [],[]
        batch = words[idx:idx+batch_size]
        for i in range(len(batch)):
            # x 是 中心词(center)
            x = batch[i]
            # y 是 周边词(context)
            y = get_target(batch , i , window_size)
            # 扩展 x和y 的纬度，因为输入是一对一对的，比如(x1, y1), (x1, y2), ...
            batch_x.extend([x]*len(y))
            batch_y.extend(y)
        # 迭代返回
        yield batch_x, batch_y

