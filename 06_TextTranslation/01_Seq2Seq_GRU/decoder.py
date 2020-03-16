'''
Decoder 解码
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device, SOS_token, EOS_token, maxlen



''' 编码器模型 '''
class DecoderGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(DecoderGRU, self).__init__()
        # 词嵌入的维度
        self.embedding_dim = embedding_dim
        # 隐藏层的维度
        self.hidden_dim = hidden_dim
        # 最大的限度为10
        self.maxlen = maxlen

        # 词嵌入函数 (当前输出是下一步输入)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        # GRU, 定义 input 和 hidden layer 的维度
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        # 当前的输出
        self.out = nn.Linear(hidden_dim, output_dim)
        # softmax 函数, 输出概率最大的那个
        self.softmax = nn.LogSoftmax(dim=1)


    # 前向传播更新 hidden_layer
    def forward(self, seq_input, hidden_layer):
        # seq_input 为 解码 的输入序列
        output = self.embedding(seq_input).view(1, 1, -1)
        # 为啥还要 relu 一下, 而在编码环节不用呢
        output = F.relu(output)
        output, hidden_layer = self.gru(output, hidden_layer)
        output = self.softmax(self.out(output[0]))
        return output, hidden_layer


    # 在进行预测时需要用到该函数进行解码
    def sample(self, pre_hidden):
        inputs = torch.tensor([SOS_token], device=device)
        hidden_layer = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            # 也是调用的 forward 函数
            output, hidden_layer = self(inputs, hidden_layer)
            # topk 是 softmax 的 属性, 返回 key 和 value
            topv, topi = output.topk(1)
            # 判断解码是否结束
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            inputs = topi.squeeze().detach()
        return res