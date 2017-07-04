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
    # counter +=1
    # if counter > 10000:
    #     break
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

documents = df_web["class_txt"]
model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=12)
model.save("model_doc2vec")






