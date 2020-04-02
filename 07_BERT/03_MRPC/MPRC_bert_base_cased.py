'''
MPRC: (Microsoft Research Paraphrase Corpus)
由微软发布，判断两个给定句子，是否具有相同的语义，属于句子对的文本二分类任务

SST(The Stanford Sentiment Treebank)，
是斯坦福大学发布的一个情感分析数据集，主要针对电影评论来做情感分类，因此SST属于单个句子的文本分类任务
（其中SST-2是二分类，SST-5是五分类，SST-5的情感极性区分的更细致）
'''


import tensorflow as tf
import tensorflow_datasets
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig, glue_convert_examples_to_features, glue_processors


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
EPOCHS = 3
PRETRAIN_PATH = '../../Data/07_BERT/02_Huggingface/02_bert-base-cased'


# 指定 MRPC 任务
TASK = 'mrpc'
if TASK == 'sst-2':
    TFDS_TASK = 'sst2'
elif TASK == 'sts-b':
    TFDS_TASK = 'stsb'
else:
    TFDS_TASK = TASK


# num_labels is 2 (0 and 1)
num_labels = len(glue_processors[TASK]().get_labels())

# Load tokenizer and model from pretrained model/vocabulary.
# Specify the number of labels to classify (2+: classification, 1: regression)
config = BertConfig.from_pretrained(PRETRAIN_PATH, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH + '/vocab.txt')
model = TFBertForSequenceClassification.from_pretrained(PRETRAIN_PATH, config=config)

# Load dataset via TensorFlow Datasets
data, info = tensorflow_datasets.load(f'glue/{TFDS_TASK}', with_info=True)
# train_examples is 3668
train_examples = info.splits['train'].num_examples
# valid_examples is 408
valid_examples = info.splits['validation'].num_examples


'''
*** train_dataset Example (128-dim) ***
guid: 2896
input_ids: 101 1103 1617 1248 3861 2686 1274 112 189 1511 3736 1121 1412 2053 1120 3254 4163 4426 119 102 1103 1214 118 2403 2849 1202 1136 1511 3736 1121 3254 4163 4426 2775 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
attention_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
token_type_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
label: 1 (id = 1)
'''
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task=TASK)
train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)


'''
*** valid_dataset Example (128-dim) ***
guid: 499
input_ids: 101 3444 7130 126 119 126 1110 1907 2786 1107 1103 10280 2231 1105 1169 7971 117 1111 170 2547 3945 1104 1164 109 1367 117 5689 119 102 3444 7130 126 119 126 1110 1208 1907 1107 1103 190 119 188 119 1105 1169 7971 1194 17599 7301 4964 1671 7995 1231 25421 1116 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
attention_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
token_type_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
label: 0 (id = 0)
'''
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task=TASK)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)


# 优化器
# epsilon: 大或等于0的小浮点数, 防止除0错误;
# clipnorm: 用于对梯度进行裁剪
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# 损失函数
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 准确率
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# 将优化器, 损失函数, 准确率放进模型内
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
train_steps = train_examples // BATCH_SIZE
valid_steps = valid_examples // EVAL_BATCH_SIZE

# 训练
history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=train_steps,
                    validation_data=valid_dataset, validation_steps=valid_steps)