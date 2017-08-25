import json
import random
import re
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



def get_vocab(file_str):
    vocab = set()
    with open(file_str, 'r') as file_:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            des = line['des']
            des = clean_up_txt(des)
            des_list = des.split()
            for word in des_list:
                vocab.add(word)
    return vocab


def vocab_overlap(files):
    for file_str in files:
        print("making vocab")
        vocab = get_vocab(file_str)
        print("writing file {}".format(file_str))
        output = open(file_str+".vocab_clean", 'w')
        with open(file_str, 'r') as file_:
            for line in file_:
                line = line.strip()
                line = json.loads(line)
                buffer_ = []
                web_txt = line['web']
                web_txt = clean_up_txt(web_txt)
                for word in web_txt.split():
                    if word not in vocab:
                        buffer_.append(word)
                web = " ".join(buffer_)
                line['web'] = web
                write_json_line(line, output)


def web_noise(files):
    noise = 0.8
    print(noise)
    for file_str in files:
        print("writing file {}".format(file_str))
        output = open(file_str+".noise{}".format(int(noise*10)), 'w')
        with open(file_str, 'r') as file_:
            for line in file_:
                line = line.strip()
                line = json.loads(line)
                web_list = line['web'].split()
                web_list = sample(web_list, noise)
                web = " ".join(web_list)
                line['web'] = web

                des_list = line['des'].split()
                des_list = sample(des_list, noise)
                des = " ".join(des_list)
                line['des'] = des

                write_json_line(line, output)



def filter_data(files):
    for file_str in files:
        print("writing file {}".format(file_str))
        output = open(file_str+".filter", 'w')
        with open(file_str, 'r') as file_:
            for line in file_:
                line = line.strip()
                line = json.loads(line)
                line['web'] = clean_up_txt(line['web'])
                line['des'] = clean_up_txt(line['des'])
                write_json_line(line, output)


#####
# What to do :
#
# Read training, and see what classes you use, then sample one more negative
#
####
def add_negative_data():
    class_descriptions = read_descriptions()
    companies_descriptions= read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
    data_path = "/home/ioannis/data/recovery_test/"
    files =[data_path +"fold{}/".format(i)+"training.json" for i in range(0,5)]
    for file_str in files:
        output = open(file_str + ".ratio2", 'w')
        with open(file_str, 'r') as file_:
            for line in file_:
                line = line.strip()
                line = json.loads(line)
                if line["class"] == "entailment":
                    write_json_line(line, output)
                else:








if __name__=="__main__":
    # data_path = "/home/ioannis/data/recovery_test/"
    # files =[data_path +"fold{}/".format(i)+"ranking_validation.json" for i in range(0,5)]
    # print(files)
    # filter_data(files)
    # web_noise(files)
    # vocab_overlap(files)

    add_negative_data()

