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
    with open("../data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            if len(txt.split()) <=MAX_DES_LEN:
                class_descriptions[class_num] = txt
    return class_descriptions


def read_meta():
    with open("../data/web_site_meta.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            id_, class_num , txt = line.split('\t')

if __name__=="__main__":
    read_descriptions()
    read_meta()


# web_class = {}
# id_txt = {}
# id_class = {}
# max_des = 0
# des_lens = []
