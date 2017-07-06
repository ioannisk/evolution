import tensorflow as tf
import pickle

lda_data = pickle.load(open("../models/topics/lda_data.pckl","rb"))
des_data = pickle.load(open("../models/topics/des_vectors.pckl","rb"))
web_data = pickle.load(open("../models/topics/web_vectors.pckl","rb"))


lda_vectors = lda_data['vectors']
lda_labels = lda_data['labels']

des_vec = des_data['vectors']
des_labels = des_data['labels']

# print(lda_data)
web_vec = web_data['vectors']
web_labels = web_data['labels']


print(len(des_vec[0]))


x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(name='wh1',shape=[784, 128])
