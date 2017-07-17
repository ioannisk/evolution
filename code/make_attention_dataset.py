import pickle
import pandas
import json
import random
import time
from utilities import data_pipeline
#### data -> dictionairy -> json


# df_web = pickle.load( open("../data/df_web.pkl","rb"))
# des_df = pickle.load(open("../data/des_df.pkl","rb"))



def wrt_dataframes():
    des_df, df_web = data_pipeline()
    with open("../data/descriptions_data.txt","w") as file_:
        for txt, class_num in zip(des_df["txt"], des_df["class_num"]):
            file_.write("{0}\t{1}\n".format(class_num, txt))
    with open("../data/web_site_data.txt", "w") as file_:
        for txt, class_num, id_ in zip(df_web["class_txt"], df_web["class_num"],df_web["company_id"]):
            if txt is not "":
                txt_buffer = ""
                for i in txt.split():
                    print i
                    kfkvnf
                # if txt.split()
                file_.write("{0}\t{1}\t{2}\n".format(id_, class_num, txt))


def read_data():
    des = {}
    web_class = {}
    id_txt = {}
    id_class = {}
    max_des = 0
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
                if len(txt.split()) <=3000:
                    # if max_des <= len(txt.split()):
                    #     max_des = len(txt.split())
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
            positive.append({'des':des_txt, 'web':web_txt, 'class':"entailment"})
    file_training =  open("../data/training_pairs.json", 'wb')
    file_validation =  open("../data/validation_pairs.json", 'wb')
    counter =0
    for i in positive:
        counter +=1
        if counter >= int(len(ids)*0.95):
            json.dump(i, file_validation)
            file_validation.write('\n')
        else:
            json.dump(i, file_training)
            file_training.write('\n')
    counter = 0
    for i in range(len(positive)):
        counter +=1
        if counter % 1000 ==0:
            print counter
        # take one random sample (sample is return as a list so take 0 element)
        id_ = random.sample(ids, 1)[0]
        unsuccesful_sample = True
        while unsuccesful_sample:
            cl = random.sample(classes,1)[0]
            if id_class[id_]==cl:
                unsuccesful_sample = True
            else:
                unsuccesful_sample = False
        negative.append({'des':des[cl], 'web':id_txt[id_], 'class':"contradiction"})
    counter = 0
    for i in negative:
        counter +=1
        if counter >= int(len(ids)*0.95):
            json.dump(i, file_validation)
            file_validation.write('\n')
        else:
            json.dump(i, file_training)
            file_training.write('\n')


# def make_training_validation():
#     training_data = []
#     validation_data = []
#     with open("../data/positive_pairs.txt", 'rb') as file_:




if __name__ =="__main__":
    wrt_dataframes()
    des, web_class, id_txt, id_class  = read_data()
    make_pairs(des, web_class, id_txt, id_class)
    # make_training_validation()

