from utilities import data_pipeline, vectorize_corpus

des_df, df_web = data_pipeline()
des_data = list(des_df["txt"])
des_labels = list(des_df["class_num"])
web_sites = list(df_web["class_txt"])
labels = list(df_web["class_num"])

vec_des_data, vec_web_sites = vectorize_corpus(des_data, web_sites,tfidf)

