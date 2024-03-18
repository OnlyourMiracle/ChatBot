# -*- coding: utf-8 -*- 
# weibo-400w

import jieba
import torch
import re
import logging
jieba.setLogLevel(logging.INFO) #关闭jieba输出信息

corpus_post = '/content/drive/MyDrive/raw_chat_corpus/weibo-400w/stc_weibo_train_post' #未处理的对话数据集
corpus_response = '/content/drive/MyDrive/raw_chat_corpus/weibo-400w/stc_weibo_train_response' #未处理的对话数据集
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") #分词处理正则
unknown = '</UNK>' #unknown字符
eos = '</EOS>' #句子结束符
sos = '</SOS>' #句子开始符
padding = '</PAD>' #句子填充负
max_voc_length = 10000 #字典最大长度
min_word_appear = 10 #加入字典的词的词频最小值
max_sentence_length = 50 #最大句子长度
save_path = '/content/drive/MyDrive/raw_chat_corpus/weibo-400w/stc_weibo_train.pth' #已处理的对话数据集保存路径

def preprocess():
    print("preprocessing...")
    '''处理对话数据集'''
    data = []
    combined_lines = []  # 初始化用于保存配对行的列表
    with open(corpus_post, 'r', encoding='utf-8') as f1, open(corpus_response, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            combined_lines.append([line1.strip(), line2.strip()])  # 移除行尾的换行符，并将两行作为列表保存

    for line in combined_lines:
        sentences = []
        for value in line:
            sentence = cop.sub("", value).split()
            #sentence = jieba.lcut(cop.sub("", value))
            sentence = sentence[:max_sentence_length] + [eos]
            sentences.append(sentence)
        data.append(sentences)

    '''生成字典和句子索引'''
    word_nums = {} #统计单词的词频
    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None
        return fun
    lambda_ = update(word_nums)
    _ = {lambda_(word) for sentences in data for sentence in sentences for word in sentence}
    #按词频从高到低排序
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)
    #词典最大长度: max_voc_length 最小单词词频: min_word_appear
    words = [word[1] for word in word_nums_list[:max_voc_length] if word[0] >= min_word_appear]
    #注意: 这里eos已经在前面加入了words,故这里不用重复加入
    words = [unknown, padding, sos] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    ix_corpus = [[[word2ix.get(word, word2ix.get(unknown)) for word in sentence]
                        for sentence in item]
                        for item in data]

    '''
    保存处理好的对话数据集

    ix_corpus: list, 其中元素为[Question_sequence_list, Answer_seqence_list]
    Question_sequence_list: e.g. [word_ix, word_ix, word_ix, ...]

    word2ix: dict, 单词:索引

    ix2word: dict, 索引:单词

    '''
    clean_data = {
        'corpus': ix_corpus, 
        'word2ix': word2ix,
        'ix2word': ix2word,
        'unknown' : '</UNK>',
        'eos' : '</EOS>',
        'sos' : '</SOS>',
        'padding': '</PAD>',
    }
    torch.save(clean_data, save_path)
    print('save clean data in %s' % save_path)

if __name__ == "__main__":
    preprocess()