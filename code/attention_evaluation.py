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
        if ["class"]=="entailement":
            training_corpus.append(web_txt)
    return training_corpus
    # return des_txt, web_txt, binary_class, des_class, web_class, web_id



def load_datasets():
    print("Loading data sets")
    with open("/home/ioannis/evolution/data/meta_training_111.json","rb") as file_:
        training_corpus = make_training_corpus(file_)
        print(len(training_corpus))


            # training_corpus.append(line[2])
    with open("/home/ioannis/evolution/data/descriptions_data.txt","rb") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            training_corpus.append(line[1])
    with open("/home/ioannis/evolution/data/meta_validation_111.json","rb") as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_data_file(file_)



if __name__=="__main__":
    load_datasets()
# def make_evaluation():

