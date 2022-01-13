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
sdate = datetime.date(year=2021, month=1, day=1)
edate = datetime.date(year=2021, month=8, day=16)

# desjardins_mc.plot_prediction(start_date=sdate, days=365, show=True)
desjardins_mc.plot_prediction_compare(start_date=sdate, sim_date=edate, days=45, show=True)
# desjardins_mc.plot_prediction(start_date=sdate, sim_date=edate, days=200, show=True)
# desjardins_mc.plot_prediction(start_date=sdate, sim_date=edate, days=365, show=True)

# desjardins_mc.plot_prediction_compare(start_date=sdate, sim_date=edate, show=True)

avg = accounts.get_daily_average(year=2021)
avg = accounts.get_daily_average()

sdate = datetime.date(year=2021, month=1, day=1)
edate = datetime.date(year=2021, month=12, day=31)

# avg2 = accounts.get_data_range_daily_average(start_date=sdate, end_date=edate)
# capital_one.get_data_by_date_range(start_date=sdate, end_date=edate)

accounts.barplot(year=2021, average=True, show=True)
for i in [10, 11, 12]:
    accounts.barplot(year=2021, month=i, show=True)
