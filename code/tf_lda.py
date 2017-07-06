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
LEARNING_RATE = 0.0001
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
# Training
########################################################

for i in range(EPOCHS):
    print("epoch {0}".format(i))
    epoch_cost = 0.0
    for j in range(0,len(data),BATCH_SIZE):
        train_x = des_vec[j:j+BATCH_SIZE]
        train_y = lda_vectors[j:j+BATCH_SIZE]
        _, cost = sess.run([optimizer, loss], feed_dict={x:train_x, y:train_y})
        epoch_cost += cost
    print(epoch_cost/len(data))






















