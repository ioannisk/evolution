import json
vocab = set()
with open("/home/ioannis/data/snli_1.0/snli_1.0_train.jsonl", "r") as file_:
    for line in file_:
        line = jsno.loads(line.strip())
        sen1 = line['sentence1']
        sen2 = line['sentence2']
        data = sen1 + " "+ sen2
        for word in data.split():
            vocab.add(word)
print(len(vocab))
