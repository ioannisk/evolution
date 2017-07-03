from evolutionai import StorageEngine
import pandas as pd
from collections import Counter
from itertools import dropwhile

N = 10

df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
count_companies = Counter()
for id_ in df["vertical"]:
    count_companies[id_] +=1
sorted_companies = sorted(count_companies.items(), key=lambda pair: pair[1], reverse=True)
more_than_dic = {}
for i in sorted_companies:
    if sorted_companies[i] >= N:
        more_than_dic[i] = sorted_companies[i]
    # print(i[0], i[1])
print(len(more_than_dic))
