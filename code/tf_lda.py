import tensorflow as tf
import pickle
import numpy as np

########################################################
# Data importing
########################################################
lda_data = pickle.load(open("../models/topics/lda_data.pckl","rb"))
des_data = pickle.load(open("../models/topics/des_vectors.pckl","rb"))
web_data = pickle.load(open("../models/topics/web_vectors.pckl","rb"))

lda_vectors = lda_data['vectors']
lda_labels = lda_data['labels']

des_vec = des_data['vectors'].todense()
des_labels = des_data['labels']

web_vec = web_data['vectors'].todense()
web_labels = web_data['labels']


data = list(zip(des_vec, lda_vectors))

########################################################
# Tensorflow model
########################################################
LEARNING_RATE = 0.5
BATCH_SIZE = 10
EPOCHS = 30

voc_size = des_vec.shape[1]
lda_topics = lda_vectors.shape[1]


x = tf.placeholder(tf.float32, [None, voc_size])
y = tf.placeholder(tf.float32,[None,lda_topics])

W = tf.get_variable(name='W',shape=[voc_size, lda_topics])
b = tf.get_variable(name='b', shape=[1,lda_topics])
pred = tf.matmul(x,W) + b

loss = tf.reduce_sum(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
########################################################


for i in range(EPOCHS):
    print("epoch {0}".format(i))
    for j in range(0,len(data),BATCH_SIZE):
        mini_batch = data[j:j+BATCH_SIZE]
        train_x, train_y = zip(*mini_batch)
        print(train_x.shape)
        print(train_y.shape)
        # print(mini_batch[0])
        # print(mini_batch[1])
        rvrv





















