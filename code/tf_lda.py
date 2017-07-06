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
print(web_data.keys())

for i in web_data:
    print i

web_vec = web_data['vectors']
web_labels = web_data['labels']
