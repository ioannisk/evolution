from utilities import data_pipeline, vectorize_corpus
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

########################################
# df_web: "class_num","class_txt","summaries","company_id","urls"
# des_df: "class_num", "class_txt", "json", "txt"
########################################

def train_naive_bayes(tfidf=False):
    des_df, df_web = data_pipeline()
    des_data = list(des_df["txt"])
    des_labels = list(des_df["class_num"])
    web_sites = list(df_web["class_txt"])
    labels = list(df_web["class_num"])
    print("vectorise")
    vec_des_data, vec_web_sites = vectorize_corpus(des_data, web_sites,tfidf)
    a=0.3 if tfidf else 0.1
    gnb = MultinomialNB(alpha=a)
    clf = gnb.fit(vec_des_data, des_labels)
    y_pred_test = clf.predict(vec_web_sites)
    print("NB Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( labels,y_pred_test, normalize=True),a))


if __name__=="__main__":
    train_naive_bayes(tfidf=False)
