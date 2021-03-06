from evolutionai import StorageEngine
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
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


def count_overlap(str1, str2):
    stopWords = set(stopwords.words('english'))
    str1=str1.split()
    str2 = str2.split()
    overlap = Counter()
    for word1 in set(str1):
        if word1 not in stopWords:
            for word2 in str2:
                if word1==word2:
                    overlap[word1] +=1
    return overlap


## META
## TITLE and DESCRIPTION

def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
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
des_df["txt"] = des_data
des_df["new_l"] = des_labels

# for i,row in des_df.iterrows():
#     print(row["class_num"], row["txt"])
# stop

df = df[df["label_num"].isin(used_classes)]


web_sites = []
labels = []
summaries = []
company_id = []
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
        # summaries.append(re.sub('\s+',' ',page_txt))
        summaries.append(page_txt)
        page_txt = clean_up_txt(page_txt)
        web_sites.append(page_txt)
        company_id.append(c_id)
        labels.append(l)
    except:
        pass
print("Vectorize documents")
df_web = pd.DataFrame()
df_web["class_num"] = labels
df_web["class_txt"] = web_sites
df_web["summaries"] = summaries
df_web["company_id"]= company_id
print("Labeled websites are {0}".format(len(df_web)))


company_dic = {}
for i, j in zip(df_web["summaries"],df_web["company_id"]):
    company_dic[j] = i

des_dic = {}
for i, j in zip(des_df["class_num"],des_df["txt"]):
    des_dic[i] = j


# wrong_web = open("wrong_web","r")
# [label pred company_id url]
df_wrong = pd.read_csv("wrong_web.txt", sep=" ")

# print(df_wrong[])
# print(wrong)

stopWords = stopwords.words('english')
# print(stopWords)
# print(type(stopWords))
# stop
while True:
    print("#####################")
    print("#####################")
    var = input("Enter website id: ")
    try:
        row = df_wrong.loc[df_wrong['company_id'] == (var)]
        # print(row)
        label =int(row["label"])
        pred = int(row["pred"])
        true_c = des_dic[label]
        pred_c = des_dic[pred]
        comapny_txt = company_dic[var]
        printable_txt = comapny_txt
        comapny_txt = clean_up_txt(comapny_txt)
        true_over = count_overlap(true_c, comapny_txt)
        pred_over = count_overlap(pred_c, comapny_txt)


        print("---------------------------------")
        print("True class is: {0}".format(label))
        print("---------------------------------")
        print(true_c)
        print("---------------------------------")
        print("Pred class is: {0}".format(pred))
        print("---------------------------------")
        print(pred_c)
        print("---------------------------------")
        print("---------------------------------")
        print(printable_txt)

        true_buffer = ""
        for key in true_over:
            true_buffer += " {0}:{1} ".format(key, true_over[key])
        print(("True {0} overlap is" + true_buffer).format(label))

        pred_buffer = ""
        for key in pred_over:
            pred_buffer += " {0}:{1} ".format(key, pred_over[key])
        print(("Pred {0} overlap is" + pred_buffer).format(pred))

    # print(dic[var])
    except:
        print("Code not in DB. Please try again")
        pass





