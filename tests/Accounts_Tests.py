#%%
import os
import Accounts
from Accounts import DesjardinsOP, DesjardinsMC, DesjardinsPR, VisaPP, CapitalOne
from Accounts import *

# ==============
# Desjardins OP
# ==============
desjardins_op.get_data_by_date()
desjardins_op.get_summed_average()
d2020 = desjardins_op.get_summed_average(year=2020)
d2021 = desjardins_op.get_summed_average(year=2021, month=7)
sns.barplot(x='total', y=d2020.index, data=d2020)
plt.show()
sns.barplot(x='total', y=d2021.index, data=d2021)
plt.show()

# desjardins_op.barplot(year=2021, month=5, show=True)
# desjardins_op.barplot(year=2021, month=6, show=True)
# desjardins_op.barplot(year=2021, month=7, show=True)
desjardins_op.get_predicted_balance()

# ==============
# Desjardins MC
# ==============
desjardins_mc.get_data_by_date()
desjardins_mc.plot_prediction(show=True)
# desjardins_mc.plot(year=2021, show=True, figsize=(15, 15))
# desjardins_mc.plot(year=2021, month=5, show=True)
# desjardins_mc.plot(year=2021, month=6, show=True)
# desjardins_mc.plot(year=2021, month=7, show=True)
# desjardins_mc.plot(year=2021, show=True, figsize=(15, 15))
desjardins_mc.get_predicted_balance()

# ==============
# Desjardins PR
# ==============
desjardins_pr.get_data_by_date()
desjardins_pr.get_predicted_balance()

# ==============
# Desjardins PrePaid VISA
# ==============
visapp.get_data_by_date()
visapp.get_predicted_balance()

# ==============
# Capital One
# ==============
visapp.get_data_by_date()
visapp.get_predicted_balance()

# ==============
# PayPal
# ==============
paypal.get_data_by_date()
paypal.get_predicted_balance()
