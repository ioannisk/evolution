import json
import random
# 1. add noise to web 20 %
# 1. add noise to web 50 %
# 2. remove all keywords of descriptions from websites
# 3. add noise to both 20 %
# 4. add noise to both 50 %

def write_json_line(json_ ,file_):
    json.dump(json_ , file_)
    file_.write('\n')

def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
    page_txt = re.sub('\s+',' ',page_txt)
    # page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    page_txt = re.sub('[^a-zA-Z]+', " ", page_txt)
    return page_txt

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





def web_noise(files):
    for file_str in files:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            web_list = line['web'].split()
            web_list = sample(web_list, 0.4)
            print('='*80)
            print(line['web'])
            web = " ".join(web_list)
            print(web)
            line['web'] = web







if __name__=="__main__":
    data_path = "/home/ioannis/data/recovery_test/"
    files =[data_path +"fold{}/".format(i)+"ranking_validation.json" for i in range(0,3)]
    web_noise(files)

