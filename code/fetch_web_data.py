from evolutionai import StorageEngine
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from collections import defaultdict
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from parse_descriptions import read_descriptions
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
# import matplotlib
# matplotlib.use('GTK')
# import matplotlib.pyplot as plt


## META
## TITLE and DESCRIPTION

def clean_up_txt(page_txt):
    page_txt = re.sub('\s+',' ',page_txt)
    page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    return page_txt


#################################
#
# Statistics about popularity of classes
#
#################################
def n_most_popular_classes(N):
    d = defaultdict(int)
    norm = 0
    # count how many times each class appears
    for i in df['label_num']:
        norm += 1
        d[i]+=1
    # number of class at pos 0 name in counts at position 1
    classes = [(key, d[key])for key in d]
    # sort classes according to popularity
    classes.sort(key=lambda tup: tup[1], reverse=True)
    total_percentage = 0
    list_of_n_classes = []
    list_of_n_classes_txt = []
    independent_percentages = []
    ## make dictionairy that given the class number it return the class name
    class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
    for i in range(N):
        total_percentage += classes[i][1]*100/norm
        independent_percentages.append(classes[i][1]*100/norm)
        # print(class_hash[classes[i][0]], classes[i][1]*100/norm)
        list_of_n_classes.append(classes[i][0])
        list_of_n_classes_txt.append(class_hash[classes[i][0]])
    top_n_classes = zip(list_of_n_classes, independent_percentages)
    return (top_n_classes, total_percentage)
# print(cksum)

#################################
#
# Remove exclusions from descriptions so text can be used as training data
#
#################################
def get_descriptions_data(des_df):
    des_data = []
    for des_json in des_df['json']:
        valid_txt = ""
        for key in des_json:
            if key!="excludes":
                valid_txt += " "+des_json[key][0]
        valid_txt = clean_up_txt(valid_txt)
        des_data.append(valid_txt)
    return des_data


def find_intersection_of_classes():
    # ["class_num", "class_txt", "json"]
    classes_desc = set(des_df["class_num"])
    classes_web = set(df["label_num"])
    intersection = classes_desc.intersection(classes_web)
    return intersection


# English stopwords
stopWords = stopwords.words('english')
# get path to the database
storage = StorageEngine("/nvme/webcache/")
# read the domains.tsv file in pandas
print("Read domains")
df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
des_df = read_descriptions()
# remover unlabelled domains
df = df[df.vertical != "None Supplied"]
label_num, label_txt = [], []
# parse the label number and label txt seperatly
for ver in df["vertical"]:
    label_n, label_t = ver.split("-",1)
    label_num.append(int(label_n))
    label_txt.append(label_t)
df["label_num"] = label_num
df["label_txt"] = label_txt
# Keep only the descriptions that exist in the dataset
intersection =find_intersection_of_classes()
des_df = des_df[des_df["class_num"].isin(intersection)]

# Companies that are not in the descriptions
#
# class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
# for i in set(df["label_num"]):
#     if i not in intersection:
#         print(i, class_hash[i])
df = df[df["label_num"].isin(intersection)]

#########################
#
# Find the classes that have detail or inclusion
#
########################
des_data = []
des_labels = []
used_classes = set()
for des_json, cl_num in zip(des_df['json'], des_df["class_num"]):
    valid_txt = ""
    ###
    ### BUG WITH OR CONDINTION
    ##
        # for key in des_json.keys():
        #     if type(des_json[key])==list:
        #         text_buffer = " "
        #         for bullet in des_json[key]:
        #             text_buffer += " " + bullet
        #     else:
        #         text_buffer = des_json[key]
        #     print("Key {0}:::::: {1}".format(key, text_buffer))

        # stop
    if ("detail" in des_json.keys()) or ("includes" in des_json.keys()) :
        used_classes.add(cl_num)
        for key in des_json:
            # print("Key: {0} ---- DES {1} ".format(key, des_json[key]))
            if key!="excludes":
                if type(des_json[key])==list:
                    text_buffer = " "
                    for bullet in des_json[key]:
                        text_buffer += " " + bullet
                else:
                    text_buffer = des_json[key]
                # text_buffer = " "
                # for bullet in des_json[key]:
                #     text_buffer += " " + bullet
                valid_txt += " "+text_buffer
        valid_txt = clean_up_txt(valid_txt)
        des_data.append(valid_txt)
        des_labels.append(cl_num)
des_df = des_df[des_df["class_num"].isin(used_classes)]
df = df[df["label_num"].isin(used_classes)]

# for da, la in zip(des_data,des_labels):
#     print("Label {0}\nData\n{1}\n\n\n".format(la,da))

# stop
###############
#
# GOOD CANDIDATE CLASSES
#
###############
# print(len(set(df["label_num"])) == len(set(des_df["class_num"])))
# focus_classes_file = open("focus_classes_noex.txt", 'w' )
# info = ['detail', 'title', 'includes']
# focus_classes = []
# focus_txt = []
# for des_json, cl_num in zip(des_df['json'], des_df["class_num"]):
#     # use flag to determine if a key in json is not in the class
#     flag = True
#     class_txt = ""
#     for i in info:
#         if i not in  des_json.keys():
#             flag = False
#         else:
#             class_txt += " " + str(des_json[i])
#     if flag:
#         focus_classes.append(cl_num)
#         focus_txt.append(class_txt)
# print(len)
# print(focus_classes)
# class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
# for i,j in zip(focus_classes,focus_txt):
#     focus_classes_file.write("{0} \t {1}\t{2}\n".format(i, class_hash[i],len(j)))
#########################
#########################
#########################

web_sites = []
labels = []
summaries = []
company_id = []
all_urls = []
print("Fetch websites from database")
counter = 0
for i, l, c_id in zip(df['url'], df["label_num"], df["company_id"]):
    # counter +=1
    # if counter > 10000:
    #     break
    # query database and get page object
    # print(i)
    page = storage.get_page(i)
    print(page)
    # some domains are not scrapped
    try:
        page_txt = page.textSummary
        summaries.append(re.sub('\s+',' ',page_txt))
        page_txt = clean_up_txt(page_txt)
        web_sites.append(page_txt)
        company_id.append(c_id)
        labels.append(l)
        all_urls.append(i)
    except:
        pass
print("Vectorize documents")
df_web = pd.DataFrame()
df_web["class_num"] = labels
df_web["class_txt"] = web_sites
df_web["summaries"] = summaries
df_web["company_id"]= company_id
df_web["urls"]=all_urls
print("Labeled websites are {0}".format(len(df_web)))
for i in range(20,100,20):
    classes, prcntg = n_most_popular_classes(i)
    print(i, prcntg)

# stop
# plt.bar(d.keys(), d.values(), width=1.0, color='g')

#################################
#
# Baseline Train on Descriptions, test on websites
#
#################################
selected_classes = {27900, 33120, 86101, 26200, 32500, 72110}

print("TRAIN ON ALL DESCRIPTIONS, TEST ON ALL WEB")
# des_labels = [i for i in des_df["class_num"]]
# , ngram_range=(1,2)
vec = CountVectorizer( min_df=1 ,stop_words=stopWords)
vec.fit(des_data)
tfidf_vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)
tfidf_vec.fit(des_data)
# print(vec.vocabulary_ == tfidf_vec.vocabulary_)
# print(tfidf_vec.idf_)


# vec_des_data = vec.transform(des_data)
# vec_web_sites = vec.transform(web_sites)

tfidf_vec_des_data = tfidf_vec.transform(des_data)
tfidf_vec_web_sites = tfidf_vec.transform(web_sites)
vec_des_data = tfidf_vec_des_data
vec_web_sites = tfidf_vec_web_sites



# print(vec.vocabulary_)
print("Desc shape {0}".format(vec_des_data.shape))
print("Web shape {0}".format(vec_web_sites.shape))

# best alpha for tfidf sublinear0.201

# best alpha is 0.11 for 1 grams with acc 0.06
# best alpha is 0.078 for 2 grams with acc 0.053
# for a in np.arange(0.001,1,0.05):
a = 0.3
gnb = MultinomialNB(alpha=a)
clf = gnb.fit(vec_des_data, des_labels)
y_pred_test = clf.predict(vec_web_sites)
# score = clf.score(vec_web_sites, labels)
# print("Score {0}".format(score))
print("NB Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( labels,y_pred_test),a))

# stop

# for c in np.arange(0.0001,1,0.05):
c = 0.0001
logistic = LogisticRegression(C=c)
clf = logistic.fit(vec_des_data, des_labels)
y_pred_test = clf.predict(vec_web_sites)
print(" LogisticTesting accuracy des - web: {0} with c {1}".format(accuracy_score( labels,y_pred_test),c))

# for c in np.arange(0.0001,2,0.5):
c = 0.0001
logistic = LinearSVC(C=c)
clf = logistic.fit(vec_des_data, des_labels)
y_pred_test = clf.predict(vec_web_sites)
print(" SVM accuracy des - web: {0} with c {1}".format(accuracy_score( labels,y_pred_test),c))


# stop
# ÷



### WRITE MISTAKES OF CLASSES OF INTERST
# wrong_web = open("wrong_web.txt", 'w' )
# wrong_web.write("label pred company_id url\n")
# for l, pred, c_id,url_ in zip(labels,y_pred_test,df_web["company_id"],df_web["urls"]):
#     if l in selected_classes and (l!=pred):
#         wrong_web.write("{0} {1} {2} {3}\n".format(l, pred, c_id,url_))
# stop

## Gather websites and descriptions that are in the top 150 classes
print("TRAIN ON TOP 150 DESCRIPTIONS, TEST ON ALL WEB")
class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
N_CLASSES = 150
top_n_prc_classes, total_percentage = n_most_popular_classes(N_CLASSES)

# class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
# for i, j in top_n_prc_classes:
#     print(class_hash[i], j)

top_n_classes, prc_top_n_classes  = zip(*top_n_prc_classes)
top_n_classes = set(top_n_classes)
prc_top_n_classes = list(prc_top_n_classes)


# Filter dataframes to inlcude only top N classes
des_df_top_n = des_df[des_df["class_num"].isin(top_n_classes)]
df_top_n = df_web[df_web["class_num"].isin(top_n_classes)]

des_data_top_n = list(des_df_top_n["class_txt"])
des_labels_top_n = list(des_df_top_n["class_num"])
web_data_top_n = list(df_top_n["class_txt"])
web_labels_top_n = list(df_top_n["class_num"])

vec = CountVectorizer( min_df=1 , stop_words=stopWords)
vec.fit(des_data_top_n)
tfidf_vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)
tfidf_vec.fit(des_data_top_n)
# print(vec.vocabulary_ == tfidf_vec.vocabulary_)
# print(tfidf_vec.idf_)


# vec_des_data = vec.transform(des_data)
# vec_web_sites = vec.transform(web_sites)

tfidf_vec_des_data = tfidf_vec.transform(des_data_top_n)
tfidf_vec_web_sites = tfidf_vec.transform(web_data_top_n)
vec_des_data = tfidf_vec_des_data
vec_web_sites = tfidf_vec_web_sites

print("Desc shape {0}".format(vec_des_data.shape))
print("Web shape {0}".format(vec_web_sites.shape))
# for a in np.arange(0.001,0.3,0.01):
a = 0.12
gnb = MultinomialNB(alpha=a)
clf = gnb.fit(vec_des_data, des_labels_top_n)
y_pred_test = clf.predict(vec_web_sites)
print("Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( web_labels_top_n,y_pred_test ),a))



    # matrix = confusion_matrix(labels, y_pred_test)
    # matrix = normalize(matrix, axis=1, norm='l1')
    # print(matrix)
# vec = CountVectorizer( min_df=1 , stop_words=stopWords)
# vec.fit(web_sites)
# vec_des_data = vec.transform(des_data)
# vec_web_sites = vec.transform(web_sites)[10000:]
# print(len(web_sites))
# # print(vec.vocabulary_)
# print(vec_des_data.shape)
# for a in np.arange(0.00001,0.3,0.0005):
#     gnb = MultinomialNB(alpha=a)
#     clf = gnb.fit(vec_des_data, des_labels)
#     y_pred_test = clf.predict(vec_web_sites)
#     print("Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( labels,y_pred_test ),a))



# #################################
# #
# # Train Web + Description --- Test Web
# #
# #################################
print("TRAIN ON WEB + DES, TEST ON ALL WEB")
des_labels = [i for i in des_df["class_num"]]
des_web_sites = des_data + web_sites
des_web_sites_labels =  des_labels + labels
data_len = len(des_web_sites)
partition = int(data_len*0.9)
# vec = CountVectorizer( min_df=1, stop_words=stopWords)

vec = CountVectorizer( min_df=1 , stop_words=stopWords)
vec.fit(des_web_sites)
tfidf_vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)
tfidf_vec.fit(des_web_sites)

des_web_sites = tfidf_vec.transform(des_web_sites)
train_X = des_web_sites[:partition]
train_y = des_web_sites_labels[:partition]
test_X = des_web_sites[partition:]
test_y =des_web_sites_labels[partition:]

print("Web shape {0}".format(train_X.shape))
# best acc is 0.207 with alpha 0.021
for a in np.arange(0.00001,0.9,0.05):
# a = 0.02201
    gnb = MultinomialNB(alpha=a )
    # data = data.toarray()
    clf = gnb.fit(train_X, train_y)
    y_pred_test = clf.predict(test_X)
    print("Testing accuracy (web + des)trainging (web) testing: {0} with alpha {1}".format(accuracy_score(test_y, y_pred_test),a ))



# #################################
# #
# # Train Web --- Test Web
# #
# #################################
print("TRAIN ON WEB, TEST ON ALL WEB")
data_len = len(web_sites)
partition = int(data_len*0.9)
vec = CountVectorizer(min_df=1,stop_words=stopWords)
data = vec.fit_transform(web_sites)
train_X = data[:partition]
train_y = labels[:partition]
test_X = data[partition:]
test_y =labels[partition:]
print("Web shape {0}".format(train_X.shape))
for a in np.arange(0.00001,0.5,0.05):
# a = 0.02001
    gnb = MultinomialNB(alpha=a)
    # data = data.toarray()
    clf = gnb.fit(train_X, train_y)
    y_pred_test = clf.predict(test_X)
    print("Testing accuracy web-web: {0} with alpha {1}".format(accuracy_score(test_y, y_pred_test), a))


