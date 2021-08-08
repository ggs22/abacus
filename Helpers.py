#%%
import datetime
from datetime import date


def print_pay_dates(start_date: datetime.date = datetime.datetime.today().date(),
                    end_date: datetime.date = datetime.datetime.today().date() + datetime.timedelta(days=90)):
    date = start_date
    for i in range (0, (end_date - start_date).days // 7):
        date = date + datetime.timedelta(days=i*14)
        if date > end_date:
            break
        # Monday == 1 ... Sunday == 7
        if date.isoweekday() == 4:
            print(f'pay,{date.year},{date.month},{date.day},0,1200,pay')


sdate = date(year=2021, month=8, day=5)
edate = date(year=2021, month=9, day=30)

print_pay_dates(start_date=sdate, end_date=edate)
