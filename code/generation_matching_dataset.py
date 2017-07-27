import pandas
import json
import random
import time
import os

MAX_LEN=111
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN


def read_descriptions():
    class_descriptions = {}
    with open("../data/descriptions_data_1.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            if len(txt.split()) <=MAX_DES_LEN:
                class_descriptions[class_num] = txt
    return class_descriptions


def read_meta():
    a = set()
    with open("../data/web_site_meta_1.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            id_, class_num , txt = line.split('\t')
            a.add(class_num)
    return a
            # if class_num not in set()

if __name__=="__main__":
    b = read_descriptions()
    a = read_meta()
    b = set(b.keys())
    print(len(b))
    print(len(a))
    print( b == a )


# web_class = {}
# id_txt = {}
# id_class = {}
# max_des = 0
# des_lens = []
