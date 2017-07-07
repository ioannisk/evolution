from utilities import data_pipeline, vectorize_corpus
from sklearn.decomposition import LatentDirichletAllocation
import pickle


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


N_TOPICS = 5
n_top_words = 20
des_df, df_web = data_pipeline()
des_data = list(des_df["txt"])
des_labels = list(des_df["class_num"])
web_sites = list(df_web["class_txt"])
labels = list(df_web["class_num"])
vec_des_data, vec_web_sites, vec = vectorize_corpus(des_data, web_sites,tfidf=False)
lda = LatentDirichletAllocation(n_topics=N_TOPICS, max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0,n_jobs=-1)
lda.fit(vec_des_data)
topics = lda.transform(vec_des_data)

lda_data = {'vectors':topics, 'labels':des_labels }
des_vectors = {'vectors':vec_des_data, 'labels':des_labels}
web_vectors = {'vectors':vec_web_sites, 'labels':labels}
pickle.dump(lda_data, open("../models/topics/lda_data.pckl","wb"))
pickle.dump(des_vectors, open("../models/topics/des_vectors.pckl","wb"))
pickle.dump(web_vectors, open("../models/topics/web_vectors.pckl","wb"))


# for i in topic:
#     print(i)
# print(topic.shape)
tf_feature_names = vec.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
