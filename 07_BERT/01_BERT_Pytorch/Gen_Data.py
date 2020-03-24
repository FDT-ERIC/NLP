'''
Thanks Tae Hwan Jung(Jeff Jung) @graykode
'''


import re

def gen_data():

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )

    # filter '.', ',', '?', '!'
    '''
    sentences = ['hello how are you i am romeo', 
                 'hello romeo my name is juliet nice to meet you', 
                 'nice meet you too how are you today', 
                 'great my baseball team won the competition', 
                 'oh congratulations juliet', 
                 'thanks you romeo']
    
    word_list = ['romeo', 'nice', 'how', 'i', 'am', 'to', 'competition', 
                 'are', 'team', 'juliet', 'today', 'too', 'oh', 'congratulations', 
                 'baseball', 'won', 'hello', 'meet', 'you', 'my', 'thanks', 'the', 
                 'great', 'is', 'name']
    
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, 
                 'oh': 4, 'team': 5, 'the': 6, 'competition': 7, 'i': 8, 'great': 9, 
                 'today': 10, 'won': 11, 'hello': 12, 'my': 13, 'is': 14, 'too': 15, 
                 'meet': 16, 'am': 17, 'you': 18, 'are': 19, 'nice': 20, 'to': 21, 
                 'congratulations': 22, 'romeo': 23, 'thanks': 24, 'how': 25, 
                 'baseball': 26, 'name': 27, 'juliet': 28}
    
    number_dict = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]', 
                   4: 'juliet', 5: 'my', 6: 'hello', 7: 'today', 8: 'meet', 
                   9: 'nice', 10: 'congratulations', 11: 'thanks', 12: 'romeo', 
                   13: 'won', 14: 'how', 15: 'i', 16: 'great', 17: 'competition', 
                   18: 'oh', 19: 'the', 20: 'am', 21: 'are', 22: 'baseball', 
                   23: 'to', 24: 'name', 25: 'too', 26: 'team', 27: 'you', 28: 'is'}
    
    token_list = [[25, 14, 21, 12, 13, 24, 28], 
                  [25, 28, 5, 16, 6, 15, 9, 18, 8, 12], 
                  [9, 8, 12, 20, 14, 21, 12, 27], 
                  [7, 5, 23, 4, 22, 10, 19], 
                  [11, 26, 15], 
                  [17, 12, 28]]
    '''

    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}

    vocab_size = len(word_dict)

    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    return sentences, word_dict, number_dict, vocab_size, token_list