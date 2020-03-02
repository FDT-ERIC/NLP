import torch
from torch import nn

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        # 为了进行矩阵相乘，改变格式
        input_vectors = input_vectors.view(batch_size, embed_size, 1)

        # Output vectors should be a batch of row vectors
        # 为了进行矩阵相乘，改变格式
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        # skip-gram 损失函数对应的第一项，最终相乘后的维度是 (batch_size * 1 * 1)
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        # 降维，变成 1 维 (batch_size * 1)
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        # 负样本损失
        # 最终相乘后的维度是 (batch_size * 5 * 1) 在 n_samples=5 的情况下
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        # sum the losses over the sample of noise vectors
        # # 降维，变成 1 维 (batch_size * 1)
        noise_loss = noise_loss.squeeze().sum(1)

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
