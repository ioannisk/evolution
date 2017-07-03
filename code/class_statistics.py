from evolutionai import StorageEngine
import pandas as pd
from collections import Counter

df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
count_companies = Counter()
for id_ in df["company_id"]:
    count_companies[id_] +=1
sorted_companies = sorted(count_companies.items(), key=lambda pair: pair[1], reverse=True)
for i in sorted_companies:
    print(i[0], i[1])
