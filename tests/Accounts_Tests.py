#%%
from Accounts import *


#%%
# for year in [2019, 2020, 2021]:
#     accounts.barplot(year=year, show=True)


#%%
# for acc in accounts:
#     print(acc.metadata.name.name)
#     print(acc.most_recent_date)

#%%
tst = desjardins_mc.get_predicted_balance()

# desjardins_op.apply_description_filter(pattern=r"[Pp][Aa][Yy][Pp][Aa][Ll]", regex=True)
# desjardins_op[0]
