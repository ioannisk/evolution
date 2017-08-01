import numpy as np
import json

def write_json_line(json_ ,file_):
    json.dump(json_ , file_)
    file_.write('\n')



# training_snli ="/home/ioannis/data/multinli_0.9/multinli_0.9_train.jsonl"
# validation_snli ="/home/ioannis/data/multinli_0.9/multinli_0.9_dev_matched.jsonl"
# testing_snli ="/home/ioannis/data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl"

# training_snli = "/home/ioannis/data/snli_1.0/snli_1.0_train.jsonl"
# validation_snli = "/home/ioannis/data/snli_1.0/snli_1.0_dev.jsonl"
testing_snli = "/home/ioannis/data/snli_1.0/snli_1.0_test.jsonl"
# files = [training_snli, validation_snli, testing_snli]
files = [testing_snli]
all_snli = []
for snli_file in files:
    with open(snli_file, 'r') as file_:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            sen1 = line["sentence1"]
            sen2 = line["sentence2"]
            label = line['gold_label']
            if line['gold_label'] == '-' or line['gold_label']=="neutral":
                continue
            json_buffer = {'des':sen1, 'web':sen2, 'class':label}
            all_snli.append(json_buffer)

with open('/home/ioannis/data/testing_binary_snli_snli.json', 'w') as file_:
    for json_buffer in all_snli:
        write_json_line(json_buffer, file_)



