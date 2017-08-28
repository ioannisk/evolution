import json
import sys
vocab = set()
# with open("/home/ioannis/data/snli_1.0/snli_1.0_train.jsonl", "r") as file_:
#     for line in file_:
#         line = json.loads(line.strip())
#         sen1 = line['sentence1']
#         sen2 = line['sentence2']
#         data = sen1 + " "+ sen2
#         for word in data.split():
#             vocab.add(word)
# print(len(vocab))
val_ids = set()
tra_ids = set()
with open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json", "r") as file_:
    for line in file_:
        line = json.loads(line.strip())
        line = line["web_id"]
        val_ids.add(line)

with open("/home/ioannis/data/recovery_test/fold2/training.json", "r") as file_:
    for line in file_:
        line = json.loads(line.strip())
        line = line["web_id"]
        val_ids.add(line)

for id_ in val_ids:
    if id_ in tra_ids:
        print("fuck")
