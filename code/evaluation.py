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

# from gensim.models import Word2Vec
# model = Word2Vec.load_word2vec_format('/home/ioannis/scp/GoogleNews-vectors-negative300.bin',binary=True)
# sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
# sentence_president = 'The president greets the press in Chicago'.lower().split()
# import nltk
# from nltk.corpus import stopwords
# stopwords = nltk.corpus.stopwords.words('english')
# sentence_obama = [w for w in sentence_obama if w not in stopwords]
# sentence_president = [w for w in sentence_president if w not in stopwords]
# distance = model.wmdistance(sentence_obama, sentence_president)

MAX_RANK = 21
RANKS = list(range(1,MAX_RANK))


# choosen_fold = "new_data_3"
# choosen_model = "new_data_3_1"
# data_file = "validation"
# data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)

# choosen_fold = "1rfolds3"
# choosen_model = "1rfolds3_1"
# data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)
###
### TOP UNSEEN CLASS FOLDS
### comment evetyhting below
choosen_fold = "recovery_test"
            # choosen_model = "best_models_1rfold3_sl"
            # choosen_fold = "recovery_test"

### prediction done for eda_models, eda_models_1, eda_models_2

choosen_model ="eda_models_3"
# choosen_model = "recovery_test"
# choosen_model = "ensemble"
data_file = "validation"
data_path = "/home/ioannis/data/{}/".format(choosen_fold)

#
# Comparison on folds 2, 4, 0
#
# folds = [0,1,2,3,4,5,6,14,15,16]
# folds = [0,1,2]
folds = [0,1,2,3,4]
# folds = [3,4]
# folds = [1]
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
    with open(data_path+"fold{}/training.json".format(fold),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            # if des_class[i] not in used_classes:
            #     continue
            X_train.append(web_txt[i])
            Y_train.append(web_class[i])
            # training_classes.add(web_class[i])
            # descriptions_txt.append(line[1])
    with open(data_path+"fold{}/{}.json".format(fold, data_file),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
        for i, b in enumerate(binary_class):
            if b!="entailment":
                continue
            # if des_class[i] not in used_classes:
            #     continue
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
            # if line[0] not in used_classes:
            #     continue
            Y_train_des.append(line[0])
            X_train_des.append(line[1])

    # X_train = X_train + X_train_des
    # Y_train = Y_train + Y_train_des
    # vec = tf_idf_vectorization(X_train)
    # # X_train_des_vec = vec.transform(X_train_des)
    # X_train_vec = vec.transform(X_train)
    # X_valid_vec = vec.transform(X_valid)
    # vec = tf_idf_vectorization(X_train_des)


    vec = count_vectorization(X_train_des)
    X_train_vec = vec.transform(X_train_des)
    Y_train = Y_train_des
    X_valid_vec = vec.transform(X_valid)
    a = 0.002
    # for a in np.arange(1,200)*0.0001:
    gnb = MultinomialNB(alpha=a,fit_prior=False)
    # clf = gnb.fit(X_train_des_vec, Y_train_des)
    clf = gnb.fit(X_train_vec, Y_train)
    y_pred_test = clf.predict(X_valid_vec)
    y_pred_train = clf.predict(X_train_vec)
    # print("Training acc is {0}".format(accuracy_score(Y_train ,y_pred_train )*100))
    # import IPython; IPython.embed()
    # print("NB Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( Y_valid,y_pred_test, normalize=True)*100,a))
    y_pred_test_proba = clf.predict_proba(X_valid_vec)
    rank_index_stats = Counter()
    true_positive = np.zeros(len(RANKS))
    for i, proba in enumerate(y_pred_test_proba):
        ranked = zip(proba, clf.classes_)
        ranked = sorted(ranked, reverse=True)
        proba, classes = zip(*ranked)
        classes = list(classes)
        # classes = remove_rare_classes(classes)
        rank_index_stats[classes.index(Y_valid[i])] +=1
        for j, TOP_N in enumerate(RANKS):
            if Y_valid[i] in classes[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(Y_valid)), rank_index_stats

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
    rank_index_stats = Counter()

    output_tf = open("fold_1_rank1_tfidf.txt", 'w')

    true_positive = np.zeros(len(RANKS))
    for i, row in enumerate(pairwise_cos_matrix):
        sim_labels = list(zip(row, des_class))
        ranked = sorted(sim_labels, reverse=True)
        similarities, classes = zip(*ranked)
        classes = list(classes)
        # classes = remove_rare_classes(classes)

        ri = classes.index(web_class[i])
        if ri ==0:
            output_tf.write("{} {}\n".format(web_class[i] , classes[ri]))

        rank_index_stats[classes.index(web_class[i])] +=1
        for j, TOP_N in enumerate(RANKS):
            if web_class[i] in classes[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(web_class)), rank_index_stats

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
            descriptions_txt.append(line[1])
    with open(data_path+"fold{}/{}.json".format(fold,data_file),"r") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
    ## train tf-idf vectorizer
    tfidf_vec = tf_idf_vectorization(descriptions_txt)
    # tfidf_vec = tf_idf_vectorization(training_corpus)
    ## vetorize des and validation websites
    des_tfidf = tfidf_vec.transform(descriptions_txt)
    web_tfidf = tfidf_vec.transform(web_txt)
    accuracy, rank_index_stats = tfidf_inference(des_tfidf, descriptions_class, web_tfidf, web_class)
    return accuracy, rank_index_stats


def decomposable_attention_eval(fold):
    # with open("/home/ioannis/evolution/entailement/multiffn-nli/src/{}/model{}/prob_predictions.txt".format(choosen_model,fold), "r") as file_:
    with open("/home/ioannis/models/{}/model{}/prob_predictions.txt".format(choosen_model,fold), "r") as file_:

    # with open("/home/ioannis/evolution/entailement/multiffn-nli/src/mnli_con_folds/model14/prob_predictions.txt".format(choosen_fold,fold), "r") as file_:
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
    rank_index_stats = Counter()
    step = len(used_classes)


    output_de = open("fold_1_rank1_de.txt", 'w')

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
        ri = used_list_des.index(list_web[0])
        if ri ==0:
            output_de.write("{} {}\n".format(list_web[0] , used_list_des[ri]))
        # print (used_list_des.index(list_web[0]))
        rank_index_stats[used_list_des.index(list_web[0])] +=1
        for j, TOP_N in enumerate(RANKS):
            if list_web[0] in used_list_des[:TOP_N]:
                true_positive[j] +=1
    return true_positive*100/float(len(companies)), rank_index_stats




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
    # nb_avrg = np.zeros(len(RANKS))
    # tfidf_avrg = np.zeros(len(RANKS))
    # att_avrg = np.zeros(len(RANKS))
    nb_avrg = np.zeros(len(folds)*len(RANKS)).reshape(len(folds), len(RANKS))
    tfidf_avrg = np.zeros(len(folds)*len(RANKS)).reshape(len(folds), len(RANKS))
    att_avrg = np.zeros(len(folds)*len(RANKS)).reshape(len(folds), len(RANKS))


    bar_nb_data = np.zeros(len(RANKS))
    bar_tf_data = np.zeros(len(RANKS))
    bar_da_data = np.zeros(len(RANKS))
    for ii, fold in enumerate(folds):
        print("###### FOLD {} ######".format(fold))

        # nb_accuracy, nb_rank_index_stats = train_naive_bayes_des_local(fold)
        # norm = float(sum(nb_rank_index_stats.values()))
        # a = sorted(nb_rank_index_stats.items())[:len(RANKS)]
        # rank_nb_probs = np.asarray(list(zip(*a))[1])/norm
        # bar_nb_data += rank_nb_probs

        # nb_avrg += nb_accuracy

        nb_accuracy, nb_rank_index_stats = train_naive_bayes_des_local(fold)
        nb_avrg[ii] = nb_accuracy
        norm = float(sum(nb_rank_index_stats.values()))
        a = sorted(nb_rank_index_stats.items())[:len(RANKS)]
        rank_nb_probs = np.asarray(list(zip(*a))[1])/norm
        bar_nb_data += rank_nb_probs

        tf_accuracy, tf_rank_index_stats = baseline_tfidf(fold)
        tfidf_avrg[ii] = tf_accuracy
        norm = float(sum(tf_rank_index_stats.values()))
        a = sorted(tf_rank_index_stats.items())[:len(RANKS)]
        rank_tf_probs = np.asarray(list(zip(*a))[1])/norm
        bar_tf_data += rank_tf_probs

        att_accuracy, da_rank_index_stats = decomposable_attention_eval(fold)
        att_avrg[ii] = att_accuracy
        norm = float(sum(da_rank_index_stats.values()))
        a = sorted(da_rank_index_stats.items())[:len(RANKS)]
        rank_da_probs = np.asarray(list(zip(*a))[1])/norm
        bar_da_data += rank_da_probs

        print_nice_table(nb_accuracy, tf_accuracy, att_accuracy)
        # print("    Decomposable attention is {}".format( accuracy))
    # for i, TOP_N in enumerate(RANKS):
    #     print("RANK {} accuracy".format(TOP_N))
    #     print("    Naive Bayes avrg {}".format(nb_avrg[i]/len(folds)))
    #     print("    TfIdf avrg {}".format(tfidf_avrg[i]/len(folds)))
    #     print("    Decomposable Attention avrg {}".format(att_avrg[i]/len(folds)))


    print(" AVERGE STATS OVER ALL FOLDS")
    plt.title('Accuracy in Top N ranks'.format(0))
    plt.ylabel('Accuracy')
    plt.xlabel('Top N')

    # plt.errorbar(x=RANKS, y=np.mean(nb_avrg,0), yerr=np.std(nb_avrg,0), label='Naive Bayes',linewidth=2, color='blue')
    # plt.plot(nb_avrg/len(folds),label='Naive Bayes',linewidth=2)
    # plt.axvline(x= np.mean(np.mean(nb_avrg,0)),linestyle='--', color='blue')

    plt.plot(np.mean(nb_avrg,0),label='Naive Bayes',linewidth=2)
    plt.fill_between(list(range(0,MAX_RANK -1)), np.mean(nb_avrg,0) - np.std(nb_avrg,0), np.mean(nb_avrg,0) + np.std(nb_avrg,0) ,alpha=0.3, facecolor='b')



    # plt.plot(tfidf_avrg/len(folds),label='Tf-idf cosine sim',linewidth=2)
    # plt.errorbar(x=RANKS,y=np.mean(tfidf_avrg,0), yerr=np.std(tfidf_avrg,0), label='Tf-idf cosine sim',linewidth=2, color='green')
    # plt.axvline(x= np.mean(np.mean(tfidf_avrg,0)),linestyle='--', color='green')

    plt.plot(np.mean(tfidf_avrg,0),label='Tf-idf cosine sim',linewidth=2)
    plt.fill_between(list(range(0,MAX_RANK -1)), np.mean(tfidf_avrg,0) - np.std(tfidf_avrg,0), np.mean(tfidf_avrg,0) + np.std(tfidf_avrg,0) ,alpha=0.3, facecolor='g')


    # plt.plot(att_avrg/len(folds),label='Decomposable Attention',linewidth=2)
    # plt.errorbar(x=RANKS,y=np.mean(att_avrg,0), yerr=np.std(att_avrg,0), label='Decomposable Attention',linewidth=2, color='red')
    # plt.axvline(x= np.mean(np.mean(att_avrg,0)),linestyle='--', color='red')

    plt.plot(np.mean(att_avrg,0),label='Decomposable Attention',linewidth=2)
    plt.fill_between(list(range(0,MAX_RANK -1)), np.mean(att_avrg,0) - np.std(att_avrg,0), np.mean(att_avrg,0) + np.std(att_avrg,0) ,alpha=0.3, facecolor='r')


    plt.legend(loc= 4)
    plt.show()
    # print([bar_nb_data/len(folds),bar_tf_data/len(folds),bar_da_data/len(folds)])
    plt.title('Accuracy in each Rank')
    xx = np.asarray(range(MAX_RANK -1))
    plt.bar(xx, bar_nb_data/len(folds), width=0.3, facecolor='b', edgecolor='b', linewidth=3, alpha=.5, label='Naive Bayes')
    plt.bar(xx+0.3, bar_tf_data/len(folds), width=0.3, facecolor='g', edgecolor='g', linewidth=3, alpha=.5, label='Tf-idf Cosine Sim')
    plt.bar(xx+0.6, bar_da_data/len(folds), width=0.3, facecolor='r', edgecolor='r', linewidth=3, alpha=.5, label='Decomposable Attention')
    plt.legend()
    plt.show()

    print_nice_table(np.mean(nb_avrg,0), np.mean(tfidf_avrg,0), np.mean(att_avrg,0))

if __name__=="__main__":
    # all_fold_stats()
    each_fold_stats()
