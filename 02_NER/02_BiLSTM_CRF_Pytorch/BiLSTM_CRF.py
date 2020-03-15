import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


''' 将 tensor 摊平成 1 维 '''
def to_scalar(var):
    '''
    :param var: variable, dim is 1
    :return: return a python float
    '''
    # return var.view(-1).tolist()[0]
    return var.view(-1).data.tolist()[0]


''' 返回横向最大值的下标 '''
def argmax(vec):
    '''
    :param vec: type is tensor
    :return: return the argmax as a python int
    '''
    _, idx = torch.max(vec, 1)  # 1 代表横向
    return to_scalar(idx)


''' 返回输入文本对应的 index 序列 '''
def prepare_sequence(seq, to_ix):
    '''
    :param seq: 输入的文本
    :param to_ix: 字典，存储着每个词对应的下标
    :return: Variable 类型
    '''
    idxs = [to_ix[w] for w in seq]  # seq 对应的 index 序列
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)  # 这一步估计是为了类型兼容吧


''' 是一个封装好的数学公式，里面先做减法的原因在于，减去最大值可以避免e的指数次，计算机上溢 '''
def log_sum_exp(vec):
    '''
    :param vec: dim is 1*5, type is Variable
    :return: 封装好的数学公式
    '''
    '''
    For Example:
    
    vec = tensor([[1,2,3,4,5]])
    
    max_score = tensor(5)
    
    max_score_broadcast = tensor([[5,5,5,5,5]])
    
    return: tensor(5.4519)
    '''
    # the dim of max_score is 1, 存储 vec 的最大值
    max_score = vec[0, argmax(vec)]
    # the dim of max_score_broadcast is 1*5
    # the dim of max_score.view(1, -1) is 1*1
    # the dim of vec.size() is 1*5
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


'''  '''
class BiLSTM_CRF(nn.Module):

    ''' 参数的定义 '''
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # input dim
        self.hidden_dim = hidden_dim  # the dim of the hidden layer
        self.vocab_size = vocab_size  # the input size of vocab
        self.tag_to_ix = tag_to_ix  # dict {tag:ix}
        self.tagset_size = len(tag_to_ix) # the number of tags

        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # Bi-LSTM 的 hidden layer
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)
        # hidden layer 到 tag 的映射
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 定义了转移矩阵, 给个随机初始化
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 不进行转移操作
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # 初始化 hidden layer
        self.hidden = self.init_hidden()


    ''' 初始化 hidden layer '''
    def init_hidden(self):
        '''
        :return:
        For Example: (2,1,3)
        tensor([[[-1.3989, -0.1273,  0.5082]],
                [[ 1.2999,  0.1419, -0.2848]]])
        '''
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim//2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim//2)))


    ''' 这个函数，只是根据 transitions 转移矩阵 ，前向传播算出的一个score，
    用到了动态规划的思想，但是因为用的是随机的转移矩阵，算出的值很大 score>20 '''
    def _forward_alg(self, feats):
        '''
        :param feats:
        :return:
        '''
        '''
        For Example:
        
        【1】 init_alphas: tensor([[-10000., -10000., -10000., 0., -10000.]])
        
        【2】 forward_var: tensor([[-10000., -10000., -10000., 0., -10000.]])
        
        【3】 feats: tensor([[1.5, 0.9, 0.1, 0.08, 0.05], 
                            [..., ..., ..., ..., ...],
                            ...
                            ]) 
                       
        【4】 emit_score: tensor([[1.5000, 1.5000, 1.5000, 1.5000, 1.5000]])
        
        【5】 trans_score: tensor([[single tag to all tags score]])
        
        【6】 alphas_t: [tensor([0.8259]), tensor([2.1739]), tensor([1.3526]), tensor([-9999.7168]), tensor([-0.7102])]
        
        【7】 forward_var: tensor([ 8.2590e-01,  2.1739e+00,  1.3526e+00, -9.9997e+03, -7.1020e-01])
        
        【8】 terminal_var: tensor([[21.1036, 18.8673, 20.7906, -9982.2734, -9980.3135]])
        '''

        # 【1】 the dim of init_alphas is 1*5
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # 将 START_TAG 的位置置零
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 【2】 wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)


        # 【3】 feats 就是 Bi-LSTM 的结果
        for feat in feats:
            # alphas_t 存储当前的 word，对应的 tag 的 score
            alphas_t = []
            for next_tag in range(self.tagset_size):

                # 【4】 the dim of emit_score is 1*5
                # 取出当前 word 对应的 tag 的 score，扩展成 1*5 的格式
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                # 【5】 the dim of trans_score is 1*5
                # 取出所有 tag 转移到 当前的 tag 的转移值
                trans_score = self.transitions[next_tag].view(1, -1)

                next_tag_var = forward_var + emit_score + trans_score

                # 【6】 the len of alphas_t is 5
                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))

            # 【7】 the dim of forward_var is 5
            # 将 alphas_t 的值合并到一个 tensor 的格式里
            forward_var = torch.cat(alphas_t).view(1, -1)

        # 【8】 最后只将最后一个 word 的 forward_var与转移 STOP_TAG 的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        # alpha 是个 0 维的 tensor 值，如 tensor(1.314)
        return alpha


    # 执行 Bi-LSTM 获取 feats
    def _get_bilstm_features(self, sentence):
        # 初始化隐藏层
        self.hidden = self.init_hidden()
        # 将输入的句子进行 词嵌入 操作，获得向量
        embeds = self.word_embeds(sentence)
        # 纵向的将 embeds 增加一个维度
        embeds = embeds.unsqueeze(1)

        # 11*1*4
        bilstm_out, self.hidden = self.bilstm(embeds, self.hidden)
        # 11*4
        bilstm_out = bilstm_out.view(len(sentence), self.hidden_dim)
        # 11*5
        bilstm_feats = self.hidden2tag(bilstm_out)

        return bilstm_feats


    # 得到 gold_seq tag 的 score, 即根据真实的 label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
    def _score_sentence(self, feats, tags):
        '''
        :param feats: 上一个函数的返回值
        :param tags: 表示输入句子对应的真实的 tags 序列
        :return: score, 句子的真实的分数
        '''
        score = autograd.Variable(torch.Tensor([0]))
        # 将 START_TAG 的标签拼接到 tags序列 最前面，这样 tags 就是12个了
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])

        for i, feat in enumerate(feats):
            '''
            self.transitions[tags[i+1], tags[i]]: 实际得到的是从标签 i 到标签 i+1 的转移概率
            feat[tags[i+1]]: feat 是 step i 的输出结果，有５个值，对应 B, I, E, START_TAG, END_TAG, 取对应标签的值
            '''
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

        return score


    # viterbi 算法, 解码, 得到预测的序列, 以及序列的得分
    def _viterbi_decode(self, feats):

        # 用于存储 tag, 相当于动态规划里的状态转移矩阵
        backpointers = []

        # Initialize the viterbi in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)

        for feat in feats:

            # holds the backpointers for this step, 用于存储当前的指针
            bptrs_t = []
            # holds the viterbi variables for this step, 用于存储当前的维特比算法的变量
            viterbivars_t = []

            for next_tag in range(self.tagset_size):

                # self.transitions[next_tag]: 其他标签（B,I,E,Start,End）到标签 next_tag 的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                # 找到最大值的下标
                best_tag_id = argmax(next_tag_var)
                # 存储当前最大值来自于哪个下标
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 动态规划的重点！！！状态转移矩阵！！！
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # self.tag_to_ix[STOP_TAG]: 其他标签到 STOP_TAG 的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path. 开始解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        # 完整性检查
        assert start == self.tag_to_ix[START_TAG]
        # 把从后向前的路径正过来
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentence, tags):
        # the dim of feast is 11*5
        feats = self._get_bilstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def forward(self, sentence):
        # Follow the back pointers to decode the best path.
        bilstm_score = self._get_bilstm_features(sentence)
        # Find the best path, given the features
        score, tag_seq = self._viterbi_decode(bilstm_score)
        return score, tag_seq



START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

training_data = [
    ("the wall street journal reported today that apple corporation made money".split(),
     "B I I I O O O B I O O".split()),
    ("georgia tech is a university in georgia".split(),
     "B I O O O O B".split())
]

# 构建字典
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# 构建字典
tag_to_ix = {"B":0, "I":1, "O":2, START_TAG:3, STOP_TAG:4}

# 定义模型
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 开始训练
for epoch in range(1):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # tensor([ 15.4958]) 最大的可能的值与 根据随机转移矩阵 计算的真实值 的差
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
        # 卧槽，这就能更新啦 ? ? ? 进行了反向传播，算了梯度值。
        # debug 中可以看到，transition 的 _grad 有了值 torch.Size([5, 5])
        neg_log_likelihood.backward()
        optimizer.step()

# 验证
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent)[0]) # score
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(model(precheck_sent)[1]) # tag sequence
