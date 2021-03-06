# Working with Bag of Words
#---------------------------------------
#
# In this example, we will download and preprocess the ham/spam
#  text data.  We will then use a one-hot-encoding to make a
#  bag of words set of features to use in logistic regression.
#
# We will use these one-hot-vectors for logistic regression to
#  predict if a text is spam or ham.

import tensorflow as tf
# import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()


def get_data(): 
    """下载数据"""
    # Check if data was downloaded, otherwise download it and save for future use
    save_file_name = os.path.join('temp','temp_spam_data.csv')

    # Create directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')

    if os.path.isfile(save_file_name):
        text_data = []
        with open(save_file_name, 'r') as temp_output_file:
            reader = csv.reader(temp_output_file)
            for row in reader:
                text_data.append(row)
    else:
        zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        r = requests.get(zip_url)
        z = ZipFile(io.BytesIO(r.content))
        file = z.read('SMSSpamCollection')
        # Format Data
        text_data = file.decode()
        text_data = text_data.encode('ascii',errors='ignore')
        text_data = text_data.decode().split('\n')
        text_data = [x.split('\t') for x in text_data if len(x)>=1]
        
        # And write to csv
        with open(save_file_name, 'w') as temp_output_file:
            writer = csv.writer(temp_output_file)
            writer.writerows(text_data)

    texts = [x[1] for x in text_data]
    target = [x[0] for x in text_data]

    # Relabel 'spam' as 1, 'ham' as 0
    target = [1 if x=='spam' else 0 for x in target]
    texts = normalize_x(texts)

    return texts,target

def normalize_x(texts):
    """文本正则化处理 Normalize text""" 
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return  texts

def plt_txt_len(texts):
    """文本长度绘图 Plot histogram of text lengths""" 
    text_lengths = [len(x.split()) for x in texts]
    text_lengths = [x for x in text_lengths if x < 50]
    plt.hist(text_lengths, bins=25)
    plt.title('Histogram of # of Words in Texts')

def plt_train(train_acc_avg):
    "Plot training accuracy over time"
    plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
    plt.title('Avg Training Acc Over Past 50 Generations')
    plt.xlabel('Generation')
    plt.ylabel('Training Accuracy')
    plt.show()

# 获取数据
texts,target = get_data() 

# 划分训练集/测试集 Split up data set into train/test 
train_indices = np.random.choice(len(texts), round(len(texts)*0.9), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

# Choose max text word length at 25
sentence_size = 25 #句子长度限制，超过25
min_word_freq = 3 #词频小于3的忽略

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
# 获取词表长度 Have to fit transform to get length of unique words.
embedding_size = len([x for x in vocab_processor.transform(texts)])
# vocab_processor.transform(texts)

# 定义模型权重和偏置项 Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# 构建单位矩阵 Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))
# 构建one-hot 词向量 Text-Vocab Embedding
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0) #[0,1,1,0,...] 
x_col_sums_2D = tf.expand_dims(x_col_sums, 0) #[[0,1,1,0,...]] #向量转矩阵,为啥？

# 定义推断模型 Declare model operations
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# 定义损失函数 Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# 定义优化器 Declare optimizer 
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 定义评估函数 Prediction operation
prediction = tf.sigmoid(model_output)

# Start a graph session, Intitialize Variables 
sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

#  Start Logistic Regression
print('Starting Training Over {} Sentences.'.format(len(texts_train)))
loss_vec = []
train_acc_all = []
train_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]] 
    # print(t)

    #单个训练，批训练的差异？
    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})

    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss) 
    if (ix+1)%10==0:
        print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))
        
    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix]==np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# Get test set accuracy
print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
test_acc_all = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]
    
    if (ix+1)%50==0:
        print('Test Observation #' + str(ix+1))    
    
    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix]==np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
# plt_train(train_acc_avg)


