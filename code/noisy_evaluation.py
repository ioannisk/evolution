import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter,defaultdict
# from generation_matching_dataset import read_descriptions, read_meta, web_des_intersection
from new_generation_matching_dataset import read_descriptions, read_meta, web_des_intersection
import matplotlib.pyplot as plt
import random

RANKS = list(range(1,201))
NOISE = 0.2
# choosen_fold = "1rfolds3"
# choosen_model = "1rfolds3_1"
# data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)
###
### TOP UNSEEN CLASS FOLDS
###
choosen_fold = "best_models_1rfold3_sl"
# choosen_model = "best_models_1rfold3_sl"
# choosen_fold = "recovery_test"
choosen_model ="recovery_test"
data_file = "validation"
data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)

#
# Comparison on folds 2, 4, 0
#
# folds = [0,1,2,3,4,5,6,14,15,16]
# folds = [0,1,2]
folds = [0,1,2]
# folds = [2]
# folds = [14]
# folds = [0,2,4]
class_descriptions = read_descriptions()
companies_descriptions= read_meta()
class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
used_classes = set(class_descriptions.keys())

less_than=10
rare_classes_set = set()
classes_companies = defaultdict(list)
for id_ in companies_descriptions:
    classes_companies[companies_descriptions[id_]["class_num"]].append(id_)
counts = {}
for key in classes_companies:
    counts[key] =len(classes_companies[key])
for key in classes_companies:
    if counts[key] <= less_than:
        rare_classes_set.add(key)

#
# Decomposable attention doesnt change
# but all other models do for the best, mayve that is a good thing
#
def remove_rare_classes(ranked_list, ):
    for rare_class in rare_classes_set:
        ranked_list.remove(rare_class)
    return ranked_list
    # return rare_classes_set





def sample(str_, rate):
    list_ = str_.split()
    buffer_ = []
    for i in list_:
        sample = random.uniform(0,1)
        if sample > rate:
            buffer_.append(i)
    buffer_ = " ".join(buffer_)
    return buffer_


def count_vectorization(corpus):
    vec = CountVectorizer( min_df=1)
    vec.fit(corpus)
    return vec


def tf_idf_vectorization(corpus):
    # print("tfidf Vectorization")
    vec = TfidfVectorizer( min_df=1)
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
    # print("cosine similarity inference")
    inference = []
    # print("des vectors {}".format(des_tfidf.shape))
    # print("web vectors {}".format(web_tfidf.shape))
    # print(len(des_class))
    pairwise_cos_matrix  = cosine_similarity(web_tfidf, des_tfidf)
    # print pairwise_cos_matrix.shape
    # print("pairwise evaluation {}".format(pairwise_cos_matrix.shape))
    assert pairwise_cos_matrix.shape == (web_tfidf.shape[0], des_tfidf.shape[0])
    true_positive = np.zeros(len(RANKS))
    for i, row in enumerate(pairwise_cos_matrix):
        sim_labels = list(zip(row, des_class))
        ranked = sorted(sim_labels, reverse=True)
        similarities, classes = zip(*ranked)
        classes = list(classes)
        # classes = remove_rare_classes(classes)
        for j, TOP_N in enumerate(RANKS):
            if web_class[i] in classes[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(web_class))

def baseline_tfidf(fold):
    # print("Loading data sets")
    descriptions_txt = []
    descriptions_class = []
    with open(data_path+"fold{}/training.json".format(fold),"r") as file_:
        training_corpus = make_training_corpus(file_)
        # print(len(training_corpus))
    with open("/home/ioannis/evolution/data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            ## ensure only used classes are used for inference
            # if line[0] not in used_classes:
            #     continue
            descriptions_class.append(line[0])
            training_corpus.append(line[1])
            descriptions_txt.append(sample(line[1],NOISE))
    with open(data_path+"fold{}/{}.json".format(fold,data_file),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
    ## train tf-idf vectorizer
    buffer__ = []
    for txt in web_txt:
        buffer__.append(sample(txt, NOISE))
    web_txt = buffer__
    tfidf_vec = tf_idf_vectorization(descriptions_txt)
    # tfidf_vec = tf_idf_vectorization(training_corpus)
    ## vetorize des and validation websites
    des_tfidf = tfidf_vec.transform(descriptions_txt)
    web_tfidf = tfidf_vec.transform(web_txt)
    accuracy = tfidf_inference(des_tfidf, descriptions_class, web_tfidf, web_class)
    return accuracy


def train_naive_bayes_des_local(fold):
    X_train =[]
    X_valid =[]
    Y_train =[]
    Y_valid =[]
    Y_train_des = []
    X_train_des = []
    training_classes = set()
    validation_classes = set()
    ids_ = set()
    with open(data_path+"fold{}/training.json".format(fold),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            X_train.append(web_txt[i])
            Y_train.append(web_class[i])
    with open(data_path+"fold{}/{}.json".format(fold, data_file),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            X_valid.append(sample(web_txt[i], NOISE))
            Y_valid.append(web_class[i])

    with open("/home/ioannis/evolution/data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            Y_train_des.append(line[0])
            X_train_des.append(sample(line[1],NOISE))

    vec = count_vectorization(X_train_des)
    X_train_vec = vec.transform(X_train_des)
    Y_train = Y_train_des
    X_valid_vec = vec.transform(X_valid)
    a = 0.002
    gnb = MultinomialNB(alpha=a,fit_prior=False)
    clf = gnb.fit(X_train_vec, Y_train)
    y_pred_test = clf.predict(X_valid_vec)
    y_pred_train = clf.predict(X_train_vec)
    y_pred_test_proba = clf.predict_proba(X_valid_vec)
    true_positive = np.zeros(len(RANKS))
    for i, proba in enumerate(y_pred_test_proba):
        ranked = zip(proba, clf.classes_)
        ranked = sorted(ranked, reverse=True)
        proba, classes = zip(*ranked)
        classes = list(classes)
        # classes = remove_rare_classes(classes)
        for j, TOP_N in enumerate(RANKS):
            if Y_valid[i] in classes[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(Y_valid))

def nb_noisy(fold):
    with open(data_path+"fold{}/ranking_validation.json".format(fold), "r") as file_:
        companies = set()
        description_class = []
        web_class = []
        counter = 0
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            web_class.append(line['web_class'])
            description_class.append(line['des_class'])
            companies.add(line['web_id'])
    true_positive = np.zeros(len(RANKS))
    step = len(used_classes)
    for i in range(0,len(web_class), step):
        print(web_class[i:i+step])


def decomposable_attention_eval(fold):
    # with open("/home/ioannis/evolution/entailement/multiffn-nli/src/{}/model{}/prob_predictions.txt".format(choosen_model,fold), "r") as file_:
    # with open("/home/ioannis/evolution/entailement/multiffn-nli/src/mnli_con_folds/model14/prob_predictions.txt".format(choosen_fold,fold), "r") as file_:

    # with open("/home/ioannis/models/{}/model{}/prob_predictions.txt".format(choosen_model,fold), "r") as file_:
    with open("/home/ioannis/models/{}/model{}/prob_predictions_noise{}.txt".format(choosen_model,fold, int(NOISE*10)), "r") as file_:



        predictions = []
        for line in file_:
            line = line.strip()
            predictions.append(float(line))
        # print(len(predictions))
    with open(data_path+"fold{}/ranking_validation.json".format(fold), "r") as file_:
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
    true_positive = np.zeros(len(RANKS))
    step = len(used_classes)
    for i in range(0,len(predictions), step):
        list_pred = predictions[i:i+step]
        list_web = web_class[i:i+step]
        # print(list_web)
        # stop
        list_des = description_class[i:i+step]
        ranked_list = sorted(list(zip(list_pred, list_web, list_des)),reverse=True)
        list_pred,list_web,list_des = zip(*ranked_list)
        list_des = list(list_des)
        # ensure only used classes are taken into consideration
        used_list_des = [jj for jj in list_des if jj in used_classes]
        # print(used_list_des[:TOP_N])
        # used_list_des.remove('87200')
        # used_list_des.remove('82990')
        # used_list_des = remove_rare_classes(used_list_des)
        for j, TOP_N in enumerate(RANKS):
            if list_web[0] in used_list_des[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(companies))




def all_fold_stats():
    nb_avrg = np.zeros(len(RANKS))
    tfidf_avrg = np.zeros(len(RANKS))
    att_avrg = np.zeros(len(RANKS))
    for fold in folds:
        print("loading fold {}".format(fold))
        # print("FOLD {} ranks".format(fold))
        accuracy = train_naive_bayes_des_local(fold)
        nb_avrg += accuracy
        # print("    Naive Bayes baseline is {}".format(accuracy))
        accuracy = baseline_tfidf(fold)
        tfidf_avrg +=accuracy
        # print("    Tf-idf baseline is {}".format(accuracy))
        accuracy = decomposable_attention_eval(fold)
        att_avrg += accuracy
        # print("    Decomposable attention is {}".format( accuracy))
    for i, TOP_N in enumerate(RANKS):
        print("RANK {} accuracy".format(TOP_N))
        print("    Naive Bayes avrg {}".format(nb_avrg[i]/len(folds)))
        print("    TfIdf avrg {}".format(tfidf_avrg[i]/len(folds)))
        print("    Decomposable Attention avrg {}".format(att_avrg[i]/len(folds)))

def print_each_fold_stats(accuracy, message):
    print("Algorithm: {}".format(message))
    for acc, ra in zip(accuracy, RANKS):
        print("Rank {} accuracy {}".format(ra, acc))

def print_nice_table(list1, list2, list3):
    print("Naive Bayes | Tf-IDF | Attention")
    for i,j in enumerate(list1):
        print("    {:.3f}      |   {:.3f}   |     {:.3f}     ".format(j, list2[i], list3[i]))



def each_fold_stats():
    nb_avrg = np.zeros(len(RANKS))
    tfidf_avrg = np.zeros(len(RANKS))
    att_avrg = np.zeros(len(RANKS))
    for fold in folds:
        print("###### FOLD {} ######".format(fold))

        nb_accuracy = train_naive_bayes_des_local(fold)
        nb_avrg += nb_accuracy

        tf_accuracy = baseline_tfidf(fold)
        tfidf_avrg +=tf_accuracy

        att_accuracy = decomposable_attention_eval(fold)
        att_avrg += att_accuracy

        print_nice_table(nb_accuracy, tf_accuracy, att_accuracy)
        # print("    Decomposable attention is {}".format( accuracy))
    # for i, TOP_N in enumerate(RANKS):
    #     print("RANK {} accuracy".format(TOP_N))
    #     print("    Naive Bayes avrg {}".format(nb_avrg[i]/len(folds)))
    #     print("    TfIdf avrg {}".format(tfidf_avrg[i]/len(folds)))
    #     print("    Decomposable Attention avrg {}".format(att_avrg[i]/len(folds)))
    print(" AVERGE STATS OVER ALL FOLDS")
    plt.title('Noisy inputs with {} rate'.format(NOISE))
    plt.ylabel('Accuracy')
    plt.xlabel('Top N')
    plt.plot(nb_avrg/len(folds),label='Naive Bayes',linewidth=2)
    plt.plot(tfidf_avrg/len(folds),label='Tf-idf cosine_similarity',linewidth=2)
    plt.plot(att_avrg/len(folds),label='Decomposable Attention',linewidth=2)
    plt.legend(loc= 4)
    plt.show()
    print_nice_table(nb_avrg/len(folds), tfidf_avrg/len(folds), att_avrg/len(folds))

if __name__=="__main__":
    each_fold_stats()
    # nb_noisy(0)

