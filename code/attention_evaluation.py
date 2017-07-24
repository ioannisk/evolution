import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TOP_N = 10

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

def load_json_validation_file(file_):
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

def tfidf_inference(des_tfidf, des_class, web_tfidf, web_class):
    true_positive = 0
    print("cosine similarity inference")
    inference = []
    print("des vectors {}".format(des_tfidf.shape))
    print("web vectors {}".format(web_tfidf.shape))
    # print(len(des_class))
    pairwise_cos_matrix  = cosine_similarity(web_tfidf, des_tfidf)
    # print pairwise_cos_matrix.shape
    print("pairwise evaluation {}".format(pairwise_cos_matrix.shape))
    assert pairwise_cos_matrix.shape == (web_tfidf.shape[0], des_tfidf.shape[0])
    for i, row in enumerate(pairwise_cos_matrix):
        sim_labels = list(zip(row, des_class))
        ranked = sorted(sim_labels, reverse=True)
        similarities, classes = zip(*ranked)
        if web_class[i] in classes[:TOP_N]:
            true_positive +=1
    return true_positive*100/float(len(web_class))

def baseline_tfidf():
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
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
    ## train tf-idf vectorizer
    # tfidf_vec = tf_idf_vectorization(descriptions_txt)
    tfidf_vec = tf_idf_vectorization(training_corpus)
    ## vetorize des and validation websites
    des_tfidf = tfidf_vec.transform(descriptions_txt)
    web_tfidf = tfidf_vec.transform(web_txt)
    accuracy = tfidf_inference(des_tfidf, descriptions_class, web_tfidf, web_class)
    return accuracy

def decomposable_attention_eval():
    with open("/home/ioannis/evolution/entailement/multiffn-nli/src/my_model_111/prob_predictions.txt", "rb") as file_:
        predictions = []
        for line in file_:
            line = line.strip()
            predictions.append(float(line))
        print len(predictions)
    with open("/home/ioannis/evolution/data/meta_ranking_validation_111.json", "rb") as file_:
        companies = set()
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            web_class = line['web_class']
            description_class = line['description_class']
            web_id = line['web_id']
            companies.add(web_id)
        print len(companies)
            # 'des':description_txt , 'web':website_txt , 'class':class_buffer, 'web_id':id_, 'web_class':web_class[i], 'des_class':description_class



if __name__=="__main__":
    accuracy = baseline_tfidf()
    print("Tf-idf baseline in top {} ranks is {}".format(TOP_N, accuracy))
    decomposable_attention_eval()

