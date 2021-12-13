#%%
from Accounts import *


#%%
# for year in [2019, 2020, 2021]:
#     accounts.barplot(year=year, show=True)


#%%
for acc in accounts:
    print(acc.metadata.name.name)
    print(acc.most_recent_date)
    print(acc.get_current_balance())

#%%
desjardins_mc.plot_prediction(start_date=datetime.date(year=2021, month=10, day=1), show=True)
desjardins_mc.plot_prediction(start_date=datetime.date(year=2021, month=10, day=1),
                              sim_date=datetime.date(year=2021, month=11, day=1),
                              show=True)
