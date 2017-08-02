import numpy as np
from attention_evaluation import load_json_validation_file
import json

choosen_fold = "1rfolds3_sl"
data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)


# for li
for i in range(3):
    with open(data_path+"fold{}/validation.json".format(i),"r") as file_:
        print("Analysing fold {}".format(i))
        counter = 0
        used_classes = set()
        for line in file_:
            if counter % 100000 == 0:
                print(counter)
            line = line.strip()
            line = json.loads(line)
            # des = line["des"]
            # web = line["web"]
            class_= ["web_class"]
            used_classes.add(class_)
            if len(des.split())==0 or len(web.split())==0:
                print(line[web_id])
            print(used_classes)



