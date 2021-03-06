import tensorflow as tf
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_regression


def top_nn_accuracy(indexes, train_l, test_label):
    train_l = np.asarray(train_l)
    true_positives = 0
    for i in range(len(indexes)):
        label = test_label[i]
        ind = list(indexes[i])
        top_n = set(train_l[ind])
        if label in top_n:
            true_positives +=1
    return true_positives*100/float(len(indexes))






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

NN = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 300
EPOCHS = 1000
HIDDEN = 300

voc_size = des_vec.shape[1]
lda_topics = lda_vectors.shape[1]

x = tf.placeholder(tf.float32, [None, voc_size])
y = tf.placeholder(tf.float32,[None,lda_topics])
lamb = tf.placeholder("float", None)
lr = tf.placeholder("float", None)
dropout = tf.placeholder("float", None)
# W = tf.get_variable(name='W',shape=[voc_size, lda_topics],initializer=tf.contrib.layers.xavier_initializer())
# b = tf.get_variable(name='b', shape=[1,lda_topics],initializer=tf.contrib.layers.xavier_initializer())
# pred = tf.matmul(x,W) + b
# square_error = tf.reduce_sum(tf.square(y - pred))
# regularizer = tf.nn.l2_loss(W)
# loss = square_error + lamb*regularizer


# W1 = tf.get_variable(name='W1',shape=[voc_size, HIDDEN])
W1 = tf.Variable(tf.random_uniform([voc_size,HIDDEN],0,0.01))
# W2 = tf.get_variable(name='W2',shape=[HIDDEN, lda_topics])
W2 = tf.Variable(tf.random_uniform([HIDDEN,lda_topics],0,0.01))

b1 = tf.get_variable(name='b1', shape=[1,HIDDEN])
b2 = tf.get_variable(name='b2', shape=[1,lda_topics])

h1 = tf.nn.relu(tf.matmul(x,W1) + b1)
h1 = tf.nn.dropout(h1, dropout)

# pred = tf.matmul(h1,W2) + b2
# square_error = tf.reduce_sum(tf.square(y - pred))
# regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
# loss = square_error + lamb*regularizer

pred =tf.nn.softmax(tf.matmul(h1,W2) + b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
loss = cross_entropy

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
########################################################
# Training
########################################################

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(lda_vectors, lda_labels)
nn = NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(lda_vectors)




# for l in [0, 0.0001, 0.001, 0.01, 20, 25, 50, 70, 100,200]:
for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    l = 0
    ################################
    # TF model
    ################################
    print("lambda {0}".format(l))
    for i in range(EPOCHS):
        # print("epoch {0}".format(i))
        epoch_cost = 0.0
        for j in range(0,len(data),BATCH_SIZE):
            train_x = des_vec[j:j+BATCH_SIZE]
            train_y = lda_vectors[j:j+BATCH_SIZE]
            _, cost = sess.run([optimizer, loss], feed_dict={x:train_x, y:train_y, lamb:l, lr:LEARNING_RATE,dropout:d})
            epoch_cost += cost
        # print("epoch_cost is {0}".format(epoch_cost/(len(data))))
        LEARNING_RATE *= 0.99
        if i%200 ==0:
            print("epoch {0}".format(i))
            print("cost is {0}".format(epoch_cost/len(data)))

    print("cost is {0}".format(epoch_cost/len(data)))
    tf_pred = sess.run(pred, feed_dict={x:des_vec,dropout:1})
    tf_pred_test = sess.run(pred, feed_dict={x:web_vec,dropout:1})
    # stop
    ################################
    # scikit model
    ################################


    # reg = linear_model.Ridge(alpha = l)
    # reg.fit(des_vec, lda_vectors)
    # tf_pred = reg.predict(des_vec)
    # tf_pred_test = reg.predict(web_vec)
    # rmse = np.mean(np.square(tf_pred-lda_vectors))
    # print("RMSE  {0}".format(rmse))


    dist_train, ind_train = nn.kneighbors(tf_pred)
    dist_test, ind_test = nn.kneighbors(tf_pred_test)

    train_acc= top_nn_accuracy(ind_train, lda_labels, des_labels)
    test_acc= top_nn_accuracy(ind_test, lda_labels, web_labels)

    print("TOP {0} TRAINING acc is {1}".format(NN,train_acc ))
    print("TOP {0} TESTING acc is {1}".format(NN,test_acc ))
    # for

    n_pred = clf.predict(tf_pred)
    n_pred_test = clf.predict(tf_pred_test)
    print("NB TRAINING acc {0}".format(accuracy_score( n_pred,des_labels, normalize=True)*100))
    print("NB TESTING acc {0}".format(accuracy_score( n_pred_test,web_labels, normalize=True)*100))
    print()
    # stop
# regression with L2 l=10 1-nn has accuracy of 0.85%






















