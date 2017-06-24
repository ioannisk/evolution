from evolutionai import StorageEngine
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from parse_descriptions import read_descriptions
import numpy as np
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
    for i in df['label_num']:
        norm += 1
        d[i]+=1
    # number of class at pos 0 name in txt at position 1
    classes = [(key, d[key])for key in d]
    # sort classes according to popularity
    classes.sort(key=lambda tup: tup[1], reverse=True)
    total_percentage = 0
    list_of_n_classes = []
    list_of_n_classes_txt = []
    ## make dictionairy that given the class number it return the class name
    class_hash = {num:txt for num, txt in zip(df["label_num"], df["label_txt"])}
    for i in range(N):
        total_percentage += classes[i][1]*100/norm
        # print(class_hash[classes[i][0]], classes[i][1]*100/norm)
        list_of_n_classes.append(classes[i][0])
        list_of_n_classes_txt.append(class_hash[classes[i][0]])
    return (list_of_n_classes, total_percentage)
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


# English stopwords
stopWords = stopwords.words('english')
# get path to the database
storage = StorageEngine("/nvme/webcache/")
# read the domains.tsv file in pandas
print("Read domains")
df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
# remover unlabelled domains
df = df[df.vertical != "None Supplied"]
label_num, label_txt = [], []
# parse the label number and label txt seperatly
for ver in df["vertical"]:
    label_n, label_t = ver.split("-",1)
    label_num.append(label_n)
    label_txt.append(label_t)
df["label_num"] = label_num
df["label_txt"] = label_txt


web_sites = []
labels = []
print("Fetch websites from database")
counter = 0
for i, l in zip(df['url'], df["label_num"]):
    counter +=1
    if counter > 10000:
        break
    # query database and get page object
    page = storage.get_page(i)
    # some domains are not scrapped
    try:
        page_txt = page.textSummary
        page_txt = clean_up_txt(page_txt)
        web_sites.append(page_txt)
        labels.append(l)
    except:
        pass
print("Vectorize documents")


for i in range(20,200,20):
    classes, prcntg = n_most_popular_classes(i)
    print(i, prcntg)

# stop
# plt.bar(d.keys(), d.values(), width=1.0, color='g')




# des_vec = CountVectorizer(min_df=1, stop_words=stopWords)
# des_data = des_vec.fit_transform(des_data)
# gnb = MultinomialNB()
# clf = gnb.fit(des_data, des_df["class_num"])
# des_pred = clf.predict(des_data)
# print("training acc: {0}".format(accuracy_score(des_df["class_num"], des_pred)))

# data = des_vec.transform(web_sites)
# web_pred = clf.predict(data)
# print("testing acc on websites: {0}".format(accuracy_score(labels, web_pred)))




des_df = read_descriptions()
# des_data = get_descriptions_data(des_df)
des_data = []
for des_json, cl_txt in zip(des_df['json'], des_df["class_txt"]):
    print(cl_txt)
    valid_txt = ""
    for key in des_json:
        print("Key: {0} ---- DES {1} ".format(key, des_json[key]))
        if key!="excludes":
            valid_txt += " "+des_json[key][0]
    valid_txt = clean_up_txt(valid_txt)
    des_data.append(valid_txt)
# return des_data


# def get_descriptions_data(des_df):
#     des_data = []
#     for des_json in des_df['json']:
#         valid_txt = ""
#         for key in des_json:
#             if key!="excludes":
#                 valid_txt += " "+des_json[key][0]
#         valid_txt = clean_up_txt(valid_txt)
#         des_data.append(valid_txt)
#     return des_data

#################################
#
# Baseline Train on Descriptions, test on websites
#
#################################
des_labels = [i for i in des_df["class_num"]]
# ngram_range=1
vec = CountVectorizer( min_df=1, stop_words=stopWords)
vec.fit(des_data)
des_data = vec.transform(des_data)
print(des_data.shape)
for a in np.arange(0.001,1,0.1):
    gnb = MultinomialNB(alpha=0.1)
    clf = gnb.fit(des_data, des_labels)
    vec_web_sites = vec.transform(web_sites)
    print(vec_web_sites.shape)
    y_pred_test = clf.predict(vec_web_sites)
    print("Testing accuracy des - web: {0} with alpha {1}".format(accuracy_score( labels,y_pred_test ),a))




# #################################
# #
# # Train Web + Description --- Test Web
# #
# #################################
# des_labels = [i for i in des_df["class_num"]]
# des_web_sites = des_data + web_sites
# des_web_sites_labels =  des_labels + labels
# data_len = len(des_web_sites)
# partition = int(data_len*0.9)
# vec = CountVectorizer( min_df=1,ngram_range=(1,2), stop_words=stopWords)
# des_web_sites = vec.fit_transform(des_web_sites)
# print("vectorization is over !!!!")
# train_X = des_web_sites[:partition]
# train_y = des_web_sites_labels[:partition]
# test_X = des_web_sites[partition:]
# test_y =des_web_sites_labels[partition:]
# gnb = MultinomialNB(alpha=0.1)
# # data = data.toarray()
# clf = gnb.fit(train_X, train_y)
# y_pred_test = clf.predict(test_X)
# print("Testing accuracy (web + des)trainging (web) testing: {0}".format(accuracy_score(test_y, y_pred_test)))



# #################################
# #
# # Train Web --- Test Web
# #
# #################################
print("Train Naive Bayes")
data_len = len(web_sites)
partition = int(data_len*0.9)
vec = CountVectorizer(min_df=1, ngram_range=(1,2),stop_words=stopWords)
data = vec.fit_transform(web_sites)
train_X = data[:partition]
train_y = labels[:partition]
test_X = data[partition:]
test_y =labels[partition:]
gnb = MultinomialNB(alpha=0.1)
# data = data.toarray()
clf = gnb.fit(train_X, train_y)
y_pred_test = clf.predict(test_X)
print("Testing accuracy web-web: {0}".format(accuracy_score(test_y, y_pred_test)))


