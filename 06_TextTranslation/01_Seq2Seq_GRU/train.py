'''
训练
'''
import torch
import torch.nn as nn
from torch import optim
# 数据处理
from data_gen import Vocabulary, data_gen, lan1, lan2
# 编码
from encoder import EncoderGRU
# 解码
from decoder import DecoderGRU
# 基本参数
from config import device, SOS_token, EOS_token, turns, print_every, print_loss_total, learning_rate, hidden_dim, embedding_dim


# 获取200个句子对
training_paris = data_gen()

# 编码
encoder = EncoderGRU(len(lan1), embedding_dim, hidden_dim).to(device)
# 解码
decoder = DecoderGRU(embedding_dim, hidden_dim, len(lan2)).to(device)
# 总的参数
params = list(encoder.parameters()) + list(decoder.parameters())
# 优化器, Adam 梯度下降
optimizer = optim.Adam(params, lr=learning_rate)
# 损失函数
criterion = nn.NLLLoss()


# 开始训练
for turn in range(turns):
    # 梯度清零
    optimizer.zero_grad()
    # 损失值
    loss = 0

    # 输入和输出
    X, y = training_paris[turn]
    # 输入的句子的长度
    input_length = X.size(0)
    # 输出的句子的长度
    target_length = y.size(0)

    # 初始化隐藏层
    hidden_layer = encoder.initHidden()

    # 编码, 更新 hidden_layer
    for i in range(input_length):
        # 调用的是 forward 函数
        hidden_layer = encoder(X[i], hidden_layer)

    # 初始化 解码 的 输入
    decoder_input = torch.LongTensor([SOS_token]).to(device)

    # 解码, 记录损失值
    for i in range(target_length):
        # hidden_layer 不变, 直接影响每一步生成的结果
        decoder_output, hidden = decoder(decoder_input, hidden_layer)
        # topk 是 softmax 的 属性, 返回 key 和 value
        topv, topi = decoder_output.topk(1)
        # 改变输出的格式, 跟实际值进行计算损失值
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[i])
        if decoder_input.item() == EOS_token:
            break

    # 一个句子的损失值
    print_loss_total += loss.item() / target_length
    # 打印
    if (turn+1) % print_every == 0:
        print("loss:{loss:,.4f}".format(loss=print_loss_total / print_every))
        print_loss_total = 0

    loss.backward()
    # 优化, 梯度下降更新参数
    optimizer.step()


# 测试一下, 看看效果
def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(EOS_token)
    f = encoder.sample(t)   # 编码
    s = decoder.sample(f)   # 解码
    r = [lan2.idx2word[i] for i in s]    # 根据id得到单词
    return ' '.join(r) # 生成句子

print(translate('我 怀疑 一起 打 游戏 。'))