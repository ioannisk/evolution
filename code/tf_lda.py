import tensorflow as tf
import pickle

lda_data = pickle.load(open("../models/topics/lda_data.pckl","rb"))
des_data = pickle.load(open("../models/topics/des_vectors.pckl","rb"))
web_data = pickle.load(open("../models/topics/web_vectors.pckl","rb"))


lda_vectors = lda_data['vectors']
lda_labels = lda_data['labels']

des_vec = des_data['vectors'].todense()
des_labels = des_data['labels']

# print(lda_data)
web_vec = web_data['vectors'].todense()
web_labels = web_data['labels']


print(des_vec.shape)
print(lda_vectors.shape)
for i,j in zip(des_vec, lda_vectors):
    print(i.shape, j.shape)


x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32,[None,10])

W = tf.get_variable(name='W',shape=[784, 128])
