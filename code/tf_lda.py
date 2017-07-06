import tensorflow as tf
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

# print(des_vec.shape)
# print(web_vec.shape)

########################################################
# Tensorflow model
########################################################
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
EPOCHS = 10

voc_size = des_vec.shape[1]
lda_topics = lda_vectors.shape[1]

x = tf.placeholder(tf.float32, [None, voc_size])
y = tf.placeholder(tf.float32,[None,lda_topics])
lamb = tf.placeholder("float", None)

W = tf.get_variable(name='W',shape=[voc_size, lda_topics])
b = tf.get_variable(name='b', shape=[1,lda_topics])
pred = tf.matmul(x,W) + b

square_error = tf.reduce_sum(tf.square(y - pred))
regularizer = tf.nn.l2_loss(W)
loss = square_error + lamb*regularizer

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

########################################################
# Training
########################################################
# for l in [0.0001, 0.001, 0.01, 0.1, 1]:
l = 0
print("lambda {0}".format(l))
for i in range(EPOCHS):
    # print("epoch {0}".format(i))
    epoch_cost = 0.0
    for j in range(0,len(data),BATCH_SIZE):
        train_x = des_vec[j:j+BATCH_SIZE]
        train_y = lda_vectors[j:j+BATCH_SIZE]
        _, cost = sess.run([optimizer, loss], feed_dict={x:train_x, y:train_y, lamb:l})
        epoch_cost += cost
    print(epoch_cost/len(data))

        # cost = sess.run([loss], feed_dict={x:web_vec, y:web_labels})
        # print("Testing Cost is {0}".format(cost/len(web_vec)))






















