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
start_date = datetime.date(year=2020, month=1, day=1)
sim_date = datetime.date(year=2021, month=12, day=31)
end_date = datetime.date(year=2022, month=12, day=31)

# desjardins_mc.plot_prction(start_date=start_date, end_date=end_date, show=True)
# # desjardins_mc.plot_predicedition_compare(start_date=start_date, sim_date=sim_date, end_date=end_date, show=True)

avg = accounts.get_daily_average(year=2021)
avg = accounts.get_daily_average()

start_date = datetime.date(year=2021, month=12, day=1)
end_date = datetime.date(year=2021, month=12, day=31)

accounts.barplot(year=2021, average=False, show=True)
accounts.barplot(year=2021, average=True, show=True)

accounts.barplot_date_range(start_date=start_date, end_date=end_date)
accounts.barplot_date_range(start_date=start_date, end_date=end_date, average=True)

for i in [10, 11, 12]:
    accounts.barplot(year=2021, month=i, show=True)

for i in [1]:
    accounts.barplot(year=2022, month=1, show=True)