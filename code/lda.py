from utilities import data_pipeline, vectorize_corpus
from sklearn.decomposition import LatentDirichletAllocation
from __future__ import print_function


N_TOPICS = 10

des_df, df_web = data_pipeline()
des_data = list(des_df["txt"])
des_labels = list(des_df["class_num"])
web_sites = list(df_web["class_txt"])
labels = list(df_web["class_num"])
vec_des_data, vec_web_sites = vectorize_corpus(des_data, web_sites,tfidf)
lda = LatentDirichletAllocation(n_topics=N_TOPICS, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(vec_des_data)
