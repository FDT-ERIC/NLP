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