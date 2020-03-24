'''
Thanks Tae Hwan Jung(Jeff Jung) @graykode
'''

import math
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Gen_Data import gen_data
from Config import *


# BERT Parameters
maxlen = 25     # the max length of input
batch_size = 6  # the number of sentences in each batch
max_pred = 5    # max tokens of prediction (最多预测5个token)
n_layers = 6    # the number of encoder layers
n_heads = 8     # multi-heads
d_model = 768   # the dim of input
d_ff = 764*4    # d_model*4, FeedForward dimension
d_k = d_v = 64  # dimension of Q, K, V
n_segments = 2  # the number of sentence parts


# 获取数据
sentences, word_dict, number_dict, vocab_size, token_list = gen_data()


# sample IsNext and NotNext to be same in small batch size
# IsNext 和 NotNext 的样本数量要一样
def make_batch():
    batch = []
    positive = negative = 0
    # 采样
    while positive != batch_size/2 or negative != batch_size/2:
        # 在 range(0, len(sentences)) 随机选一个数
        tokens_a_index, token_b_index = randrange(len(sentences)), randrange(len(sentences))
        # 通过上一步随机选取的数选择对应的句子
        token_a, token_b = token_list[tokens_a_index], token_list[token_b_index]
        # 输入需要加上 [CLS] 和 [SEP] 对应的 index
        input_ids = [word_dict['[CLS]']] + token_a + [word_dict['[SEP]']] + token_b + [word_dict['[SEP]']]
        # 0 或者 1
        segment_ids = [0] * (1+len(token_a)+1) + [1] * (len(token_b)+1)

        # MASK LM
        # 15 % of tokens in one sentence
        # n_pred 表示需要预测的 token 数量
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
        # [CLS] 和 [SEP] 不能作为 MASK 的候选
        cand_masked_pos = [i for i, token in enumerate(input_ids)
                           if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        # 打乱候选索引的顺序
        shuffle(cand_masked_pos)

        # MASK 操作
        # masked_pos 存放 mask 操作的索引值, masked_tokens 存放 mask 操作的具体值
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
            elif random() < 0.5:  # 20% 里面的 50%, 也就是整体的 10%
                # 替换成 vocab 里面的其他词
                index = randint(0, vocab_size-1)
                input_ids[pos] = word_dict[number_dict[index]]

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_pos.extend([0] * n_pad)
            masked_tokens.extend([0] * n_pad)

        # IsNext
        if tokens_a_index + 1 == token_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        # NotNext
        elif tokens_a_index + 1 != token_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch


# 0 填充 torch.Size([batch_size, len_q, len_k])
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


# 激活函数
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# 词嵌入层
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment embedding
        self.norm = nn.LayerNorm(d_model)  # normalization

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


# 普通的 Q, K, V 操作, 唔系好明, attn_mask 的作用是什么
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    # Q, K, V 操作, attn_mask 的作用是什么, 不太懂
    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # Fills elements of self tensor with value where mask is one.
        # 矩阵内所有 1 都替换成 -1e9
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# 基于 MultiHead 的 self-attention, 唔系好明
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model]
        # k: [batch_size x len_k x d_model]
        # v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D)
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size x n_heads x len_q x d_v]
        # attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # output: [batch_size x len_q x d_model]
        output = nn.Linear(n_heads * d_v, d_model)(context)

        return nn.LayerNorm(d_model)(output + residual), attn


# FeedForward
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


# Encoder 编码
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs to same Q, K, V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# BERT 模型
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()  # 词嵌入
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 一共有 n_layers 层 Encoder
        self.fc = nn.Linear(d_model, d_model)  # 全连接层
        self.activ1 = nn.Tanh()  # Tanh 激活函数
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        
    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, maxlen, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, len, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, maxlen, n_vocab]

        return logits_lm, logits_clsf


# 定义模型和超参
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)


# 训练
for epoch in range(100):
    optimizer.zero_grad()
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    # for masked LM
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
    loss_lm = (loss_lm.float()).mean()
    # for sentence classification
    loss_clsf = criterion(logits_clsf, isNext)
    # loss 由两部分组成
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()


# 预测
# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print([number_dict[w] for w in input_ids if number_dict[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                               torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('true isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)