#%%
from Accounts import *

# for year in [2019, 2020, 2021]:
#     accounts.barplot(year=year, show=True)
    # account.get_summed_data(year=year)
    # account.barplot(year=year, average=False, show=True)
    # account.barplot(year=year, average=True, show=True)


visapp._load_from_raw_files()

for acc in accounts:
    print(acc.metadata.name.name)
    print(acc.most_recent_date)
