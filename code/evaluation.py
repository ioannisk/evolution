import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter
from generation_matching_dataset import read_descriptions, read_meta, web_des_intersection

TOP_N = 3


folds = [14,15,16]
class_descriptions = read_descriptions()
companies_descriptions= read_meta()
class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
used_classes = set(class_descriptions.keys())


def train_naive_bayes_des_local(fold):
    # used_classes = find_only_used_classes()
    X_train =[]
    X_valid =[]
    Y_train =[]
    Y_valid =[]
    Y_train_des = []
    X_train_des = []
    training_classes = set()
    validation_classes = set()
    ids_ = set()
    with open("/home/ioannis/evolution/data/folds/fold{}/training.json".format(fold),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            if des_class[i] not in used_classes:
                continue
            X_train.append(web_txt[i])
            Y_train.append(web_class[i])
            # training_classes.add(web_class[i])
            # descriptions_txt.append(line[1])
    with open("/home/ioannis/evolution/data/folds/fold{}/validation.json".format(fold),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            if des_class[i] not in used_classes:
                continue
            X_valid.append(web_txt[i])
            Y_valid.append(web_class[i])
            # validation_classes.add(web_class[i])
    # all_classes = training_classes.union(validation_classes)
    # print(len(all_classes))

    # stop
    with open("/home/ioannis/evolution/data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            ## ensure only used classes are used for inference
            if line[0] not in used_classes:
                continue
            Y_train_des.append(line[0])
            X_train_des.append(line[1])

    # X_train = X_train + X_train_des
    # Y_train = Y_train + Y_train_des
    # vec = tf_idf_vectorization(X_train)
    # # X_train_des_vec = vec.transform(X_train_des)
    # X_train_vec = vec.transform(X_train)
    # X_valid_vec = vec.transform(X_valid)
    vec = tf_idf_vectorization(X_train_des)
    X_train_vec = vec.transform(X_train_des)
    Y_train = Y_train_des
    X_valid_vec = vec.transform(X_valid)
    a = 0.8
    # for a in np.arange(1,20)*0.1:
    gnb = MultinomialNB(alpha=a,fit_prior=False)
    # clf = gnb.fit(X_train_des_vec, Y_train_des)
    clf = gnb.fit(X_train_vec, Y_train)
    y_pred_test = clf.predict(X_valid_vec)
    y_pred_train = clf.predict(X_train_vec)
    # print("Training acc is {0}".format(accuracy_score(Y_train ,y_pred_train )*100))
    # import IPython; IPython.embed()
    # print("NB Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( Y_valid,y_pred_test, normalize=True)*100,a))
    y_pred_test_proba = clf.predict_proba(X_valid_vec)
    true_positive = 0
    for i, proba in enumerate(y_pred_test_proba):
        ranked = zip(proba, clf.classes_)
        ranked = sorted(ranked, reverse=True)
        proba, classes = zip(*ranked)
        if Y_valid[i] in classes[:TOP_N]:
            true_positive +=1
    return true_positive*100/float(len(Y_valid))


def tf_idf_vectorization(corpus):
    # print("tfidf Vectorization")
    stopWords = stopwords.words('english')
    # vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords, sublinear_tf=False)
    vec = TfidfVectorizer( min_df=1,sublinear_tf=True)
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
    return des_txt, web_txt, binary_class, des_class, web_class, web_id

def tfidf_inference(des_tfidf, des_class, web_tfidf, web_class):
    true_positive = 0
    # print("cosine similarity inference")
    inference = []
    # print("des vectors {}".format(des_tfidf.shape))
    # print("web vectors {}".format(web_tfidf.shape))
    # print(len(des_class))
    pairwise_cos_matrix  = cosine_similarity(web_tfidf, des_tfidf)
    # print pairwise_cos_matrix.shape
    # print("pairwise evaluation {}".format(pairwise_cos_matrix.shape))
    assert pairwise_cos_matrix.shape == (web_tfidf.shape[0], des_tfidf.shape[0])
    for i, row in enumerate(pairwise_cos_matrix):
        sim_labels = list(zip(row, des_class))
        ranked = sorted(sim_labels, reverse=True)
        similarities, classes = zip(*ranked)
        classes = list(classes)
        if web_class[i] in classes[:TOP_N]:
            true_positive +=1
    return true_positive*100/float(len(web_class))

def baseline_tfidf(fold):
    # print("Loading data sets")
    descriptions_txt = []
    descriptions_class = []
    with open("/home/ioannis/evolution/data/folds/fold{}/training.json".format(fold),"r") as file_:
        training_corpus = make_training_corpus(file_)
        # print(len(training_corpus))
    with open("/home/ioannis/evolution/data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            ## ensure only used classes are used for inference
            if line[0] not in used_classes:
                continue
            descriptions_class.append(line[0])
            training_corpus.append(line[1])
            descriptions_txt.append(line[1])
    with open("/home/ioannis/evolution/data/folds/fold{}/validation.json".format(fold),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
    ## train tf-idf vectorizer
    # tfidf_vec = tf_idf_vectorization(descriptions_txt)
    tfidf_vec = tf_idf_vectorization(training_corpus)
    ## vetorize des and validation websites
    des_tfidf = tfidf_vec.transform(descriptions_txt)
    web_tfidf = tfidf_vec.transform(web_txt)
    accuracy = tfidf_inference(des_tfidf, descriptions_class, web_tfidf, web_class)
    return accuracy


def decomposable_attention_eval(fold):
    with open("/home/ioannis/evolution/data/folds/fold{}/training.json".format(fold), "r") as file_:
        predictions = []
        for line in file_:
            line = line.strip()
            predictions.append(float(line))
        # print(len(predictions))
    with open("/home/ioannis/evolution/data/folds/fold{}/validation.json".format(fold), "r") as file_:
        companies = set()
        description_class = []
        web_class = []
        counter = 0
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            counter +=1
            web_class.append(line['web_class'])
            description_class.append(line['des_class'])
            companies.add(line['web_id'])
    true_positive = 0
    step = len(used_classes)
    for i in range(0,len(predictions), step):
        list_pred = predictions[i:i+step]
        list_web = web_class[i:i+step]
        list_des = description_class[i:i+step]
        ranked_list = sorted(list(zip(list_pred, list_web, list_des)),reverse=True)
        list_pred,list_web,list_des = zip(*ranked_list)
        list_des = list(list_des)
        # ensure only used classes are taken into consideration
        used_list_des = [jj for jj in list_des if jj in used_classes]
        if list_web[0] in used_list_des[:TOP_N]:
            true_positive +=1
    return true_positive*100/float(len(companies))




if __name__=="__main__":
    for fold in folds:
        print("FOLD {} ranks".format(fold))
        accuracy = train_naive_bayes_des_local(fold)
        print("    Naive Bayes baseline is {}".format(accuracy))
        accuracy = baseline_tfidf(fold)
        print("    Tf-idf baseline is {}".format(accuracy))
        accuracy = decomposable_attention_eval(fold)
        print("    Decomposable attention is {}".format(TOP_N, accuracy))
