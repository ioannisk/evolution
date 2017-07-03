from evolutionai import StorageEngine
import pandas as pd
from collections import Counter
from itertools import dropwhile

df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
count_companies = Counter()
for id_ in df["vertical"]:
    count_companies[id_] +=1
count_companies = dict(count_companies)

for key, count in dropwhile(lambda key_count: key_count[1] >= 20, count_companies.most_common()):
    del count_companies[key]


sorted_companies = sorted(count_companies.items(), key=lambda pair: pair[1], reverse=True)
for i in sorted_companies:
    print(i[0], i[1])


print(len(sorted_companies))
