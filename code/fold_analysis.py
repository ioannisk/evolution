import numpy as np
from attention_evaluation import load_json_validation_file
import json

choosen_fold = "1rfolds3_sl"
data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)


# for li
for i in range(3):
    with open(data_path+"fold{}/training.json".format(i),"r") as file_:
        for line in file_:
            line = line.strip()
            line = json.load(line)
            des = line["des"]
            web = line["web"]
            if len(des.split())==0 orlen(web.split())==0:
                print(line[web_id])



