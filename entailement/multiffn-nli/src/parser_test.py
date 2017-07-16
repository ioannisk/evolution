import nltk
import json
def test_snli(filename="/home/ioannis/data/snli_1.0"):
    with open(filename, 'rb') as f:
        line = line.strip()
        for line in f:
            data = json.loads(line)
            sentence1_parse = data['sentence1_parse']
            sentence2_parse = data['sentence2_parse']
            tree1 = nltk.Tree.fromstring(sentence1_parse)
            tree2 = nltk.Tree.fromstring(sentence2_parse)
            tokens1 = tree1.leaves()
            tokens2 = tree2.leaves()
            sen1 = data["sentence1"]
            sen2 = data["sentence2"]

            print tokens1
            print sen1
            iecnicn

if __name__=="__main__":
    test_snli()
