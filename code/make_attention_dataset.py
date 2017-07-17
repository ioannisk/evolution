import pickle
import pandas
import json
import random
#### data -> dictionairy -> json


# df_web = pickle.load( open("../data/df_web.pkl","rb"))
# des_df = pickle.load(open("../data/des_df.pkl","rb"))

def read_data():
    des = {}
    web_class = {}
    id_txt = {}
    id_class = {}
    with open("../data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            des[class_num] = txt
    with open("../data/web_site_data.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            try:
                id_, class_num , txt = line.split('\t')
                if class_num in web_class:
                    web_class[class_num].append((id_, txt))
                else:
                    web_class[class_num] = []
                    web_class[class_num].append((id_, txt))
                id_txt[id_] = txt
                id_class[id_] = class_num
            except:
                pass


    return des, web_class, id_txt, id_class


def make_pairs(des, web_class, id_txt, id_class):
    positive = []
    negative = []
    ids = set(id_txt.keys())
    classes = set(des.keys())
    for class_num in web_class:
        des_txt = des[class_num]
        for id_, web_txt in web_class[class_num]:
            positive.append((des_txt, web_txt))
    counter = 0
    for i in range(len(positive)):
        counter +=1
        if counter % 10000 ==0:
            print counter

        # take one random sample (sample is return as a list so take 0 element)
        id_ = random.sample(ids, 1)[0]
        unsuccesful_sample = True
        while unsuccesful_sample:
            cl = random.sample(classes,1)[0]
            unsuccesful_sample = False
            if id_class[id_]==cl:
                unsuccesful_sample = True
        negative.append((des[cl], id_txt[id_]))




if __name__ =="__main__":
    des, web_class, id_txt, id_class  = read_data()
    make_pairs(des, web_class, id_txt, id_class)

