import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords,vocabulary=vec.vocabulary_, sublinear_tf=True)

def tf_idf_vectorization(corpus):
    print("TF_IDF Vectorization")
    stopWords = stopwords.words('english')
    vec = TfidfVectorizer( min_df=1 ,stop_words=stopWords, sublinear_tf=True)
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

def load_json_data_file(file_):
    des_txt = web_txt = binary_class = des_class = web_class = web_id = []
    counter = 0
    for line in file_:
        line = line.strip()
        line = json.loads(line)
        des_txt.append(line["des"])
        web_txt.append(line["web"])
        binary_class.append(line["class"])
        des_class.append(line["des_class"])
        web_class.append(line["web_class"])
        web_id.append(line["web_id"])
        counter +=1
    # print counter
    print len(des_txt)
    return des_txt, web_txt, binary_class, des_class, web_class, web_id



def load_datasets():
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
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_data_file(file_)
    # print(len(des_txt))
    # print(len(web_txt))
    # print(len(binary_class))



if __name__=="__main__":
    load_datasets()
# def make_evaluation():

