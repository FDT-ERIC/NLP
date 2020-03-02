import torch
from torch import nn


# 网络结构
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        '''
        :param n_vocab: 词典的大小
        :param n_embed: 词向量的大小 (纬度)
        :param noise_dist: 负采样的单词分布，如 [0.2, 0.3, 0.5], 表示对应位置的概率
        '''
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        # Initialize embedding tables with uniform distribution
        # I believe this helps with convergence
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)


    def forward_input(self, input_words):
        '''
        :param self: 自个儿
        :param input_words: 中心词 (center), One-Hot 的形式
        :return: 返回中心词的词向量
        '''
        input_vectors = self.in_embed(input_words)
        return input_vectors


    def forward_output(self, output_words):
        '''
        :param self: 自个儿
        :param output_words: 周边词 (context),
        :return: 返回周边词的词向量
        '''
        output_vectors = self.out_embed(output_words)
        return output_vectors


    # 对负采样的处理
    """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
    def forward_noise(self, batch_size, n_samples):
        '''
        :param self: 自个儿
        :param batch_size: 一次放到内存中的数据的量
        :param n_samples: 负样本的数量
        :return:
        '''

        if self.noise_dist is None:
            # Sample words uniformly
            # 所有单词 等概率 进行采样
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist


        # Sample words from our noise distribution
        # 多项式采样 noise_dist=[0.2, 0.3, 0.5], noise_words=[2]
        # 返回的是采样的下标
        noise_words = torch.multinomial(noise_dist, batch_size * n_samples,
                                        replacement=True)

        # 找到周边词的词向量, 并改变一下格式 (batch_size, n_samples, self.n_embed), 以便后续的计算
        noise_vectors = self.out_embed(noise_words).view(batch_size,
                                        n_samples, self.n_embed)

        return noise_vectors
