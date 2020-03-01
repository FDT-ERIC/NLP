from gensim.models import Word2Vec # 词向量
from random import choice # 随机选取一个item
from os.path import exists # 判断文件是否存在
import warnings
warnings.filterwarnings('ignore') # 不打印警告信息


# 基本配置，参数
class CONF:
    path = '../../../Data/04_TextGeneration/Poem_古诗/poem.txt'  # 训练语料的路径
    window = 16  # Word2Vec划窗的大小
    min_count = 60  # 过滤低频字
    size = 125  # 词向量的纬度
    topn = 14  # 概率最大的 n 个词
    model_path = '01_poem_word2vec.model' # 词向量模型


# 诗词生成模型
class Model:
    # 启动
    def __init__(self, window, topn, model):
        '''
        :param window: 划窗大小
        :param topn: 概率最大的 topn 个词
        :param model: 词向量模型
        '''
        self.window = window
        self.topn = topn
        self.model = model
        self.chr_dict = model.wv.index2word # 字典


    """模型初始化"""
    @classmethod
    def initialize(cls, config):
        '''
        :param config: 传入配置
        :return: 不太懂 cls 这里
        '''
        # 看是否存在已经训练好的词向量模型
        if exists(config.model_path):
            # 模型读取
            model = Word2Vec.load(config.model_path)
        else:
            # 预料读取
            with open(config.path, encoding='utf-8') as f:
                be_train = [list(line.strip()) for line in f]

            model = Word2Vec(sentences=be_train, size=config.size,
                             window=config.window, min_count=config.min_count)
            model.save(config.model_path)

        # cls 是啥
        return cls(config.window, config.topn, model)


    """古诗词生成"""
    def poem_generator(self, title, form):
        '''
        :param title: 用户输入的诗词的标题
        :param form: 决定是哪种类型的诗词, 包括 ‘五言绝句’，‘七言绝句’, '对联'
        :return:
        '''

        # 过滤器，将 逗号 和 句号 过滤
        filter = lambda lst: [t[0] for t in lst if t[0] not in ['，', '。']]

        # 标题补全
        if len(title) < 4:
            # 如果没标题，则去字典里随便选择一个
            if not title:
                title += choice(self.chr_dict)
            for _ in range(4-len(title)):
                # 根据标题的最后一个字，找出最相近的那些字，组成标题
                similar_chr = self.model.similar_by_word(title[-1], self.topn // 2)
                # 过滤掉 逗号 和 句号
                similar_chr = filter(similar_chr)
                # 在 self.topn//2 这么多个相似的字中随机选取
                char = choice([c for c in similar_chr if c not in title])
                title += char

        # 文本生成
        poem = list(title)
        # 总共 form[0] 个句子
        for i in range(form[0]):
            # 每句 form[1] 个字
            for _ in range(form[1]):
                # 基于前 self.window 个字，预测下一个字
                predict_chr = self.model.predict_output_word(
                    poem[-self.window:], max(self.topn, len(poem)+1)
                )
                # 将预测出来的 逗号 和 句号 过滤掉
                predict_chr = filter(predict_chr)
                # 在预测的字中，最近选择不重复的字，最为下一个字
                char = choice([c for c in predict_chr if c not in poem[len(title):]])
                # 把预测的字加到列表里
                poem.append(char)
            # 添加标点符号
            poem.append('，' if i % 2 == 0 else '。')
        # 诗的主体的长度 为 length_main
        length_main = form[0] * (form[1] + 1)
        return '《%s》' % ''.join(poem[:-length_main]) + '\n' + ''.join(poem[-length_main:])


# main 函数，处理输入，生成，输出
def main(config=CONF):
    form = {'五言绝句': (4, 5), '七言绝句': (4, 7), '对联': (2, 9)}
    m = Model.initialize(config)
    while True:
        title = input('输入标题: ').strip()
        try:
            poem = m.poem_generator(title, form['五言绝句'])
            print('\033[031m%s\033[0m' % poem + '\n')  # 红色
            poem = m.poem_generator(title, form['七言绝句'])
            print('\033[033m%s\033[0m' % poem + '\n')  # 黄色
            poem = m.poem_generator(title, form['对联'])
            print('\033[036m%s\033[0m' % poem + '\n')  # 绿色
        except:
            pass

if __name__ == '__main__':
    main()



