'''
Encoder 编码
'''
import torch
import torch.nn as nn
from config import device



''' 编码器模型 '''
class EncoderGRU(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(EncoderGRU, self).__init__()
        # 词嵌入的维度
        self.embedding_dim = embedding_dim
        # 隐藏层的维度
        self.hidden_dim = hidden_dim

        # 词嵌入函数
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # GRU, 定义 input 和 hidden layer 的维度
        self.gru = nn.GRU(embedding_dim, hidden_dim)


    # 前向传播更新 hidden_layer
    def forward(self, input, hidden_layer):
        # GRU 的输入格式为: (seq_len, batch, input_size)
        embedded = self.embedding(input).view(1, 1, self.embedding_dim)
        output, hidden_layer = self.gru(embedded, hidden_layer)
        return hidden_layer


    # 在进行预测时需要用到该函数进行编码
    def sample(self, seq_list):
        # 放到设备上
        word_index = torch.LongTensor(seq_list).to(device)
        hidden_layer = self.initHidden()
        for word_tensor in word_index:
            # 也是调用的 forward 函数
            hidden_layer = self(word_tensor, hidden_layer)
        return hidden_layer


    # 初始化隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)