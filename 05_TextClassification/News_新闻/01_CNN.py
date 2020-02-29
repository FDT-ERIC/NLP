'''
词向量 + CNN =》 文本分类
'''
'''
时间复杂度: 高
空间复杂度: 中
'''

from warnings import filterwarnings
filterwarnings('ignore')  # 不打印警告
from numpy import argmax
from DataPreprocessing import load_xy, id2label  # 加载向量
from sklearn.metrics import classification_report  # 预测报告
# keras
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import to_categorical



# 参数
size = 100  # 词向量的长度
maxlen = 200  # 序列长度
output_dim = 100  # 词向量纬度
batch_size = 512  # 一次处理的数据大小
epochs = 99  # 循环迭代次数
verbose = 2  # 日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
patience = 2  # 参数没有进一步更新的轮数
callbacks = [EarlyStopping('val_acc', patience=patience)]
# callbacks = [EarlyStopping('val_acc', patience=patience)]
validation_split = .05  # 验证集的比例为 5%


# 词向量处理
X_train, X_test, y_train, y_test = load_xy()
# 将多个序列截断或补齐为相同长度
X_train = pad_sequences(X_train, maxlen, dtype='float')
X_test = pad_sequences(X_test, maxlen, dtype='float')
# 相当于 one-hot embedding
y_train = to_categorical(y_train, 9)
y_test = to_categorical(y_test, 9)


# CNN
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, GlobalMaxPool1D
filters = 50  # 卷积滤波器数量
kernel_size = 10  # 卷积滤波器大小
model = Sequential(name='Keras_CNN')
# (None, 200, 50)
model.add(Conv1D(filters, kernel_size*2, padding='same', activation='relu', input_shape=(maxlen, size)))
# (None, 100, 50)
model.add(MaxPool1D(pool_size=2))  # strides 默认等于 pool_size
# (None, 100, 100)
model.add(Conv1D(filters*2, kernel_size, padding='same', activation='relu'))
# (None, 100)
model.add(GlobalMaxPool1D())  # 对于时序数据的全局最大池化，维度为 (None, 100)
# (None. 9)
model.add(Dense(9, activation='softmax'))  # 全连接层
print(model.summary())


# 训练 (adam 梯度下降，crossentropy 损失函数)
model.compile('adam', 'categorical_crossentropy', ['acc'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
          callbacks=callbacks, validation_split=validation_split)

# 模型保存
model.save('W2V_CNN.h5')

# 打印测试结果
def metric(y_test, y_pred, verbose=True):
    i2l = id2label()
    y_test = [i2l[i] for i in y_test]
    y_pred = [i2l[i] for i in y_pred]
    report = classification_report(y_test, y_pred)
    print(report)

# 打印模型结果
y_pred = model.predict(X_test)
metric(argmax(y_test, axis=1), argmax(y_pred, axis=1))