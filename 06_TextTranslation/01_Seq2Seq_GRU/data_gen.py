'''
处理数据
'''
import torch
import random
from config import EOS_token, device, turns



'''构造字典 存储 word 与 id 的映射 '''
class Vocabulary(object):

    # 初始化
    def __init__(self):
        # word 到 index 的映射
        self.word2idx = {}
        # index 到 word 的映射
        self.idx2word = {0: "<SOS>", 1: "<EOS>", -1:"<unk>"}
        # 初始化为2, 因为算上了开始(<SOS>)和结束(<EOS>>
        self.idx = 2

    # 添加词到字典里
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    # 将句子的词添加到字典里
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    # 返回 word 的 index
    def __call__(self, word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]

    # 返回字典的长度
    def __len__(self):
        return self.idx



# 训练数据
data = [['你 很 聪明 。', 'you are very wise .'],
        ['我们 一起 打 游戏 。', 'let us play game together .'],
        ['你 太 刻薄 了 。', 'you are so mean .'],
        ['你 完全 正确 。', 'you are perfectly right .'],
        ['我 坚决 反对 妥协 。', 'i am strongly opposed to a compromise .'],
        ['他们 正在 看 电影 。', 'they are watching a movie .'],
        ['他 正在 看着 你 。', 'he is looking at you .'],
        ['我 怀疑 他 是否 会 来 。', 'i am doubtful whether he will come .']]


# language1, 输入的语言
lan1 = Vocabulary()
# language2, 输出的语言
lan2 = Vocabulary()

# 根据训练数据构建两个语言的 Vocab
for i, j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)


# 将句子变成其对应的索引, 格式为 tensor, 如 tensor([[1], [2], [3], [4], [5]])
def sentence2tensor(lang, sentence):
    indexes = [lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    # .view(-1, 1) 将 (1*5) [1,2,3,4,5] -> (5*1) [[1], [2], [3], [4], [5]]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# 返回一个句子对, 对应的索引序列, 格式为 tensor
def pair2tensor(pair):
    # 输入的句子
    input_tensor = sentence2tensor(lan1, pair[0])
    # 翻译的句子
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)

# 将训练数据变成相应的输入的格式
def data_gen():
    # 获取200个句子对
    training_paris = [pair2tensor(random.choice(data)) for pair in range(turns)]
    return training_paris
