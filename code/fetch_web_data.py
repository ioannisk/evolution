from evolutionai import StorageEngine
import pandas as pd

storage = StorageEngine("/nvme/webcache/")



companies_df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
companies_df['url'] = ["".join(["http://www.", u]) if u[:4] != "www." else "".join(["http://", u]) for u in companies_df['url']]

page = s.get_page(url)
print(page.links)
