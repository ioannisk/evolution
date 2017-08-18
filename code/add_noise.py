import json
import random
# 1. add noise to web 20 %
# 1. add noise to web 50 %
# 2. remove all keywords of descriptions from websites
# 3. add noise to both 20 %
# 4. add noise to both 50 %

def sample(list_, rate):
    buffer_ = []
    for i in list_:
        sample = random.uniform(0,1)
        if sample > rate:
            buffer_.append(i)
    return buffer_



def des_vocabulary(file_str):
    vocab = set()
    with open(file_str, 'r') as file_:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            des_list = line['des'].split()
            print("---")
            print(len(des_list))
            des_list = sample(des_list, 0.4)
            print(len(des_list))
            des = " ".join(des_list)
            line['des'] = des



def web_noise(files):
    for file_str in files:
        vocab = des_vocabulary(file_str)
        print(len(vocab))
        # with open(file_str, 'r') as file_:
        #     for line in file_:
        #         line = line.strip
        #         line = json.loads(line)
        #         line['web']







if __name__=="__main__":
    data_path = "/home/ioannis/data/recovery_test/"
    files =[data_path +"fold{}/".format(i)+"ranking_validation.json" for i in range(0,3)]
    web_noise(files)

