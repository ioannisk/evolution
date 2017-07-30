import numpy as np
from attention_evaluation import load_json_validation_file

choosen_fold = "folds5"
data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)

with open(data_path+"fold{}/training.json".format(fold),"r") as file_:
    des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)

