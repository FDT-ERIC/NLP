'''
参数配置
'''
import torch


# 计算装备啊, GPU啊, 工作了自己配一个
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 开始
SOS_token = 0
# 停止
EOS_token = 1

# 解码的最大长度
maxlen = 10

# 词向量维度
embedding_dim = 500
# 隐藏层
hidden_dim = 256

# 学习率
learning_rate = 0.001
# 训练 200个 句子对
turns = 200
# 每训练 20个 句子对 就打印一次信息
print_every = 20
# 记录 一个句子对 的的损失值
print_loss_total = 0