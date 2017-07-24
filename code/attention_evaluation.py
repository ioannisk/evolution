import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)

# def cosine_sim_parts_calculator(query, string):
#     q = []
#     s = []
#     q.append(query)
#     s.append(string)
#     query_v = tfidf_vectorizer.transform(q)
#     string_v = tfidf_vectorizer.transform(s)
#     # string_v = tfidf_vectorizer.transform(s)
#     return cosine_similarity(query_v, string_v)[0][0]

def tf_idf_vectorization(corpus):
    print("tfidf Vectorization")
    stopWords = stopwords.words('english')
    # vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords, sublinear_tf=False)
    vec = TfidfVectorizer( min_df=1,sublinear_tf=False)
    vec.fit(corpus)
    return vec

def make_training_corpus(file_):
    training_corpus = []
    for line in file_:
        line = line.strip()
        line = json.loads(line)
        des_txt = line["des"]
        web_txt = line["web"]
        binary_class = line["class"]
        des_class = line["des_class"]
        web_class = line["web_class"]
        web_id = line["web_id"]
        if binary_class=="entailment":
            training_corpus.append(web_txt)
    return training_corpus
    # return des_txt, web_txt, binary_class, des_class, web_class, web_id

def load_json_data_file(file_):
    des_txt, web_txt, binary_class, des_class, web_class, web_id = [], [],[], [], [], []
    for line in file_:
        line = line.strip()
        line = json.loads(line)
        if line["class"]=="entailment":
            des_txt.append(line["des"])
            web_txt.append(line["web"])
            binary_class.append(line["class"])
            des_class.append(line["des_class"])
            web_class.append(line["web_class"])
            web_id.append(line["web_id"])
            # training_corpus.append(web_txt)
    # print counter
    return des_txt, web_txt, binary_class, des_class, web_class, web_id

## find if in top N
def classification(list_, true_cl, N):
    srt = sorted(list_, key=lambda x: x[0], reverse=True)
    for i, (sim, cl) in enumerate(list_):
        if cl == true_cl:
            return 1
        if i >=649:
            return 0

###########################
### Maybe use linear kernel
###########################

def tfidf_inference(des_tfidf, des_class, web_tfidf, web_class):
    print("cosine similarity inference")
    inference = []
    print((des_tfidf.shape))
    print((web_tfidf.shape))
    print(len(des_class))
    pairwise_cos_matrix  = cosine_similarity(web_tfidf, des_tfidf)
    # print pairwise_cos_matrix.shape
    print(pairwise_cos_matrix.shape)
    assert pairwise_cos_matrix.shape = (web_tfidf.shape[0], des_tfidf.shape[1])
    # for





    stop
    for i, (web, web_cl) in enumerate(zip(web_tfidf, web_class)):
        similarities = []
        if i%100==0:
            print i
        for des, des_cl in zip(des_tfidf, des_class):
            sim = cosine_similarity(web,des)[0][0]
            similarities.append((sim, des_cl))
        inference.append(classification(similarities, web_cl, 5))
    accuracy = sum(inference)/float(len(inference))
    return accuracy


        # print document


def main():
    print("Loading data sets")
    descriptions_txt = []
    descriptions_class = []
    with open("/home/ioannis/evolution/data/meta_training_111.json","rb") as file_:
        training_corpus = make_training_corpus(file_)
    with open("/home/ioannis/evolution/data/descriptions_data.txt","rb") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            descriptions_class.append(line[0])
            training_corpus.append(line[1])
            descriptions_txt.append(line[1])
    with open("/home/ioannis/evolution/data/meta_validation_111.json","rb") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_data_file(file_)
    ## train tf-idf vectorizer
    tfidf_vec = tf_idf_vectorization(descriptions_txt)
    ## vetorize des and validation websites
    des_tfidf = tfidf_vec.transform(descriptions_txt)
    web_tfidf = tfidf_vec.transform(web_txt)
    accuracy = tfidf_inference(des_tfidf, descriptions_class, web_tfidf, web_class)
    print accuracy






if __name__=="__main__":
    main()
# def make_evaluation():

