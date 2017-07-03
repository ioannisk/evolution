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
more_than = []
for i in sorted_companies:
    if i[1] >= N:
        more_than.append(i)

for i in more_than:
    print(i[0], i[1])
print(len(more_than))
