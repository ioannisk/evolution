import nltk
import json

def replace_list(str1):
    chars = ".,!?"
    for i in chars:
        st1 = str1.replace(i, " {0}".format(i))
    return str1

def test_snli(filename="/home/ioannis/data/snli_1.0/snli_1.0_train.jsonl"):
    with open(filename, 'rb') as f:

        for line in f:
            line = line.strip()
            data = json.loads(line)
            sentence1_parse = data['sentence1_parse']
            sentence2_parse = data['sentence2_parse']
            tree1 = nltk.Tree.fromstring(sentence1_parse)
            tree2 = nltk.Tree.fromstring(sentence2_parse)
            tokens1 = tree1.leaves()
            tokens2 = tree2.leaves()
            sen1 = data["sentence1"]
            sen1 = replace_list(sen1)
            sen1 = sen1.split()
            sen2 = data["sentence2"]
            sen2 = replace_list(sen2)
            sen2 = sen2.split()

            a = (sen1 == tokens1)
            b = (sen2 == tokens2)
            if (not a) or (not b):
                print sen1
                print tokens1

            # print tokens1
            # print sen1.split()
            # iecnicn

if __name__=="__main__":
    test_snli()
