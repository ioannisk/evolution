import tensorflow as tf
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model

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
LEARNING_RATE = 0.00001
BATCH_SIZE = 50
EPOCHS = 500
HIDDEN = 100

voc_size = des_vec.shape[1]
lda_topics = lda_vectors.shape[1]

x = tf.placeholder(tf.float32, [None, voc_size])
y = tf.placeholder(tf.float32,[None,lda_topics])
lamb = tf.placeholder("float", None)

# W = tf.get_variable(name='W',shape=[voc_size, lda_topics])
# b = tf.get_variable(name='b', shape=[1,lda_topics])
# pred = tf.matmul(x,W) + b
# square_error = tf.reduce_sum(tf.square(y - pred))
# regularizer = tf.nn.l2_loss(W)
# loss = square_error + lamb*regularizer


W1 = tf.get_variable(name='W1',shape=[voc_size, HIDDEN])
W2 = tf.get_variable(name='W2',shape=[HIDDEN, lda_topics])


b1 = tf.get_variable(name='b1', shape=[1,HIDDEN])
b2 = tf.get_variable(name='b2', shape=[1,lda_topics])

h1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
pred = tf.nn.softmax(tf.matmul(h1,W2) + b2)

# square_error = tf.reduce_sum(tf.square(y - pred))
# regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
# loss = square_error + lamb*regularizer


# pred =tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
loss = cross_entropy


optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(lda_vectors, lda_labels)

########################################################
# Training
########################################################
for l in [0, 0.001, 0.01, 0.1, 1, 10, 15, 20, 25, 50]:
    # l = 0
    ################################
    # TF model
    ################################
    # print("lambda {0}".format(l))
    # for i in range(EPOCHS):
    #     # print("epoch {0}".format(i))
    #     epoch_cost = 0.0
    #     for j in range(0,len(data),BATCH_SIZE):
    #         train_x = des_vec[j:j+BATCH_SIZE]
    #         train_y = lda_vectors[j:j+BATCH_SIZE]
    #         _, cost = sess.run([optimizer, loss], feed_dict={x:train_x, y:train_y, lamb:l})
    #         epoch_cost += cost
    # print("cost is {0}".format(epoch_cost/len(data)))
    # tf_pred = sess.run(pred, feed_dict={x:des_vec})
    ################################
    # scikit model
    ################################

    reg = linear_model.Ridge (alpha = l)
    reg.fit(des_vec, lda_vectors)
    tf_pred = reg.predict(des_vec)
    tf_pred_test = reg.predict(web_vec)

    n_pred = clf.predict(tf_pred)
    n_pred_test = clf.predict(tf_pred_test)
    print("NB TRAINING acc {0}".format(accuracy_score( n_pred,des_labels, normalize=True)*100))
    print("NB TESTING acc {0}".format(accuracy_score( n_pred_test,web_labels, normalize=True)*100))
    print()

# regression with L2 l=10 1-nn has accuracy of 0.85%






















