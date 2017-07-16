import pickle
import pandas
import json
import random
#### data -> dictionairy -> json


# df_web = pickle.load( open("../data/df_web.pkl","rb"))
# des_df = pickle.load(open("../data/des_df.pkl","rb"))

def read_data():
    des = {}
    web = {}
    with open("../data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            des[class_num] = txt
    with open("../data/web_site_data.txt". "r") as file_:
        for line in file_:
            line = line.strip()
            id_, class_num , txt = line.split('\t')
            web[class_num] = txt
    return des, web


def make_pairs(des, web):
    positive = []
    for key in web



if __name__ =="__main__":
    des, web  = read_data()

