#%%
from Accounts import *

# for year in [2019, 2020, 2021]:
#     accounts.barplot(year=year, show=True)
    # account.get_summed_data(year=year)
    # account.barplot(year=year, average=False, show=True)
    # account.barplot(year=year, average=True, show=True)


for acc in accounts:
    print(acc.metadata.name.name)
    print(acc.most_recent_date)


desjardins_op.apply_description_filter(pattern=r"[Pp][Aa][Yy][Pp][Aa][Ll]", regex=True)
desjardins_op[0]
