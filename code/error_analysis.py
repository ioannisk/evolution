from evolutionai import StorageEngine
from utilities import data_pipeline
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

########################################
# df_web: "class_num","class_txt","summaries","company_id","urls"
# des_df: "class_num", "class_txt", "json", "txt"
########################################

def train_naive_bayes(tfidf=False):
    des_df, df_web = data_pipeline()
    stopWords = stopwords.words('english')
    des_data = list(des_df["txt"])
    des_labels = list(des_df["class_num"])
    web_sites = list(df_web["class_txt"])
    labels = list(df_web["class_num"])
    print("vectorise")
    vec = CountVectorizer( min_df=1 ,stop_words=stopWords)
    vec.fit(des_data)
    if tfidf == True:
        tfidf_vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)
        tfidf_vec.fit(des_data)
        tfidf_vec_des_data = tfidf_vec.transform(des_data)
        tfidf_vec_web_sites = tfidf_vec.transform(web_sites)
        vec_des_data = tfidf_vec_des_data
        vec_web_sites = tfidf_vec_web_sites
    else:
        vec_des_data = vec.transform(des_data)
        vec_web_sites = vec.transform(web_sites)
    print("Desc shape {0}".format(vec_des_data.shape))
    print("Web shape {0}".format(vec_web_sites.shape))
    a = 0.1
    gnb = MultinomialNB(alpha=a)
    clf = gnb.fit(vec_des_data, des_labels)
    y_pred_test = clf.predict(vec_web_sites)
    print("NB Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( labels,y_pred_test, normalize=True),a))


if __name__=="__main__":
    train_naive_bayes(tfidf=True)
