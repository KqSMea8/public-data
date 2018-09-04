#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本预处理

CountVectorizer/TfidfVectorizer 的局限
1. 向量特征稀疏
2. 向量顺序是词表的顺序，无法表达真实句子中单词出现的位置

词袋模型（Bag of Words）    

词嵌入(Word Embedding)
https://cloud.tencent.com/developer/article/1041795

https://blog.csdn.net/dugudaibo/article/details/79071541
正交性使得具有相似属性的词之间的关系变得微弱
相似的单词会聚集在一起，而不同的单词会分开；每个坐标轴可以看作是区分这些单词的一种属性
基于统计（Count based ）和基于预测（Perdition based）

http://www.52nlp.cn/%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%8E%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E7%AC%AC%E4%BA%8C%E8%AE%B2%E8%AF%8D%E5%90%91%E9%87%8F
共现矩阵(Cooccurrence matrix)X

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

def load_data(file_path, top_n=10):
    """ 加载fasttext格式的训练数据
    https://stackabuse.com/read-a-file-line-by-line-in-python/
    """
    labels_dataset = []
    txts_dataset = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i > top_n:
                break

            fields = line.strip().split(' ')
            max_label_index = max(
                [fi for fi, field in enumerate(fields) if '__label__' in field])

            label = [f.replace('__label__', '')
                     for f in fields[:max_label_index+1]]
            txt = ' '.join(fields[max_label_index+1:])

            labels_dataset.append(label)
            txts_dataset.append(txt)

    return labels_dataset, txts_dataset


def CountVectorizer_test(labels, txt_lines):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """
    vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    vect.fit(txt_lines)

    # 将语料中所有出现的单词放在一个词表中，vect.vocabulary_, 单词：索引编号
    # 标点符号等会丢弃，但有时标点符号等也是重要特征
    print(len(vect.vocabulary_), type(vect.vocabulary_), vect.vocabulary_)

    # 构造矩阵，词表顺序, 不存在0，存在统计出现的次数
    xtrain_v = vect.transform(txt_lines)

    # print(vect.shape)
    print(type(xtrain_v))
    # return

    line_count = len(labels)
    for i in range(line_count):
        print("----{}---".format(i))
        print(txt_lines[i])
        print(xtrain_v[i].shape)
        print(xtrain_v[i])


def TfidfVectorizer_test(labels, txt_lines):
    """
    TF(词语频率)=（该词语在文档出现的次数）/（文档中词语的总数）
    IDF(逆文档频率)= log_e（文档总数/出现该词语的文档总数）
    TF-IDF = TF * IDF 
    http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    vect = TfidfVectorizer(
        analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

    # #ngram
    # vect = TfidfVectorizer(
    #     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)

    # char？
    # vect = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

    vect.fit(txt_lines)

    print(len(vect.vocabulary_), type(vect.vocabulary_), vect.vocabulary_)

    xtrain_tfidf = vect.transform(txt_lines)

    line_count = len(labels)
    for i in range(line_count):
        print("----{}---".format(i))
        print(txt_lines[i])
        print(xtrain_tfidf[i].shape)
        print(xtrain_tfidf[i])


def load_embedding_vec(file_path='output/model_cooking_6.vec'):
    """加载词向量"""
    embeddings_index = {}
    for i, line in enumerate(open(file_path)):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    print("embeddings_index.shape={}".format(len(embeddings_index)))
    return embeddings_index

def embedding_test(txt_lines,embeddings_index):
    """词嵌入
    """
    #创建一个分词器
    token = text.Tokenizer()
    token.fit_on_texts(txt_lines)
    word_index = token.word_index
    print("word_index len={}".format(len(word_index))) #词汇表

    #将文本转换为分词序列，并填充它们保证得到相同长度的向量
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(txt_lines), maxlen=70)
    # print(train_seq_x) #为啥前面的是0，

    #创建分词嵌入映射，词向量矩阵
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("embedding_matrix.shape {}".format( embedding_matrix.shape)  )   
    print(embedding_matrix)

if __name__ == "__main__":
    labels, txt_lines = load_data('data/cooking/cooking.train', 4)
    print(txt_lines) 

    # CountVectorizer_test(labels,txt_lines)
    # TfidfVectorizer_test(labels, txt_lines)

    emb = load_embedding_vec() 
    # print(emb['apple'])
    # print(emb['oil'])
    embedding_test(txt_lines,emb)
