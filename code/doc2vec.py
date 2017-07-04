import gensim
from gensim.models.doc2vec import Doc2Vec
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
from collections import Counter
from sklearn.metrics import accuracy_score
from parse_descriptions import read_descriptions
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from os import listdir
from os.path import isfile, join


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),labels=[self.labels_list[idx]])


def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
    page_txt = re.sub('\s+',' ',page_txt)
    page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    return page_txt

# def remove_url_chars(string):
#     for ch in ['/','.', '_', '#', '-']:
#         string = string.lower()
#         if ch in string:
#             string=string.replace(ch," ")
#     return string


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
storage = StorageEngine("/nvme/webcache_old/")
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
    counter +=1
    if counter > 1000:
        break
    # query database and get page object
    page = storage.get_page(i)
    # some domains are not scrapped
    try:
        page_txt = page.textSummary

        summaries.append(re.sub('\s+',' ',page_txt))
        # some preprocessing
        page_txt = clean_up_txt(page_txt)
        # page_txt = remove_url_chars(page_txt)
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


#
#   Prepare the data
#

# for id_, txt in zip(df_web["company_id"], df_web["summaries"]):
#     f = open("doc2vec_data/{0}.txt".format(id_), "w")
#     f.write(txt)


# stop

docLabels = []
docLabels = [f for f in listdir("doc2vec_data") if f.endswith('.txt')]
data = []
for doc in docLabels:
    dd = open("doc2vec_data/" + doc, 'r')
    data.append(dd)
    dd.close()
# print(data[0])
# print(docLabels[0])
# LabeledSentence = gensim.models.doc2vec.LabeledSentence
it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    print(epoch)
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)

# documents = list(df_web["class_txt"])
# documents_raw = list(df_web["summaries"])

# for i in range(10):
#     print('------------------')
#     print('------------------')
#     print('------------------')
#     print('------------------')
#     print(documents_raw[i])
#     print('------------------')
#     print(documents[i])

# model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=12)
# model.save("model_doc2vec")






