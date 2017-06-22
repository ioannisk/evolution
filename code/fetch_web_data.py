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
# import matplotlib
# matplotlib.use('GTK')
# import matplotlib.pyplot as plt


## META

## TITLE and DESCRIPTION

def clean_up_txt(page_txt):
    page_txt = re.sub('\s+',' ',page_txt)
    page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    return page_txt

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
# web_vectorizer = CountVectorizer(min_df=1, stop_words=stopWords)
# data = web_vectorizer.fit_transform(web_sites)
# train_X = data[:129637]
# train_y = labels[:129637]
# test_X = data[:129637]
# test_y =labels[:129637]
# d = defaultdict(int)
# for i in df['label_num']:
#     d[i]+=1
# plt.bar(d.keys(), d.values(), width=1.0, color='g')



des_df = read_descriptions()
des_data = []
for des_json in des_df['json']:
    valid_txt = ""
    for key in des_json:
        if key!="excludes":
            valid_txt += " "+des_json[key][0]
    des_data.append(valid_txt)


des_vec = CountVectorizer(min_df=1, stop_words=stopWords)
des_data = des_vec.fit_transform(des_data)
gnb = MultinomialNB()
clf = gnb.fit(des_data, des_df["class_num"])
des_pred = clf.predict(des_data)
print("training error: {0}".format(accuracy_score(des_df["class_num"], des_pred)))

data = des_vec.transform(web_sites)
print(data.shape)
print(len(labels))
web_pred = clf.predict(data)
# train_X = data[:129637]
# train_y = labels[:129637]
# test_X = data[:129637]
# test_y =labels[:129637]

print("testing error on websites: {0}".format(accuracy_score(labels, web_pred)))



# Full Naive Bayes
print("Train Naive Bayes")
gnb = MultinomialNB()
# data = data.toarray()
clf = gnb.fit(train_X, train_y)
y_pred_test = clf.predict(test_X)
print(accuracy_score(test_y, y_pred_test))


