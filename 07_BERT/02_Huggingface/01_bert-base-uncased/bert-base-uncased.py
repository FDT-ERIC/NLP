import torch
from transformers import BertTokenizer, BertForMaskedLM

# token
tokenizer = BertTokenizer.from_pretrained('../../../Data/07_BERT/Huggingface/01_bert-base-uncased/vocab.txt')
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was an engineer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# 掩码一个标记，我们将尝试用' BertForMaskedLM '预测回来
masked_index = 11
tokenized_text[masked_index] = '[MASK]'

# 将标记转换为词汇表索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# segment
segments_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 将输入转换为PyTorch张量
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# 加载预训练模型 (weights)
model = BertForMaskedLM.from_pretrained('../../../Data/07_BERT/Huggingface/01_bert-base-uncased')
model.eval()

# 吧数据扔到模型里
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# 预测
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print("Predicted token is: ", predicted_token)