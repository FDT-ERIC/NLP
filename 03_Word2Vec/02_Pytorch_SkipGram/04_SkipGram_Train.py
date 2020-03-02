# from '01_DataPreprocessing' import train_words, get_batch, noise_dist, word_to_idx
# from '02_Network' import SkipGramNeg,
# from '03_NegativeSampling' import NegativeSamplingLoss
# import torch
# from torch import optim
#
#
# embedding_dim = 300
# model = SkipGramNeg(len(word_to_idx), embedding_dim, noise_dist=noise_dist)
#
# # using the loss that we defined
# criterion = NegativeSamplingLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)
#
# print_every = 1500
# steps = 0
# epochs = 5
# batch_size = 500
# n_samples = 5
#
# # train for some number of epochs
# for e in range(epochs):
#     # get our input, target batches
#     for input_words, target_words in get_batch(train_words, batch_size):
#         steps += 1
#         inputs, targets = torch.LongTensor(input_words),
#         torch.LongTensor(target_words)
#
#         # input, output, and noise vectors
#         input_vectors = model.forward_input(inputs)
#         output_vectors = model.forward_output(targets)
#         noise_vectors = model.forward_noise(batch_size, n_samples)
#
#         # negative sampling loss
#         loss = criterion(input_vectors, output_vectors, noise_vectors)
#         if steps//print_every == 0:
#             print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()