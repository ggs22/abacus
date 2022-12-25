import datetime
from utils.utils import print_lti_pays, root_dir, pickle_dir, data_dir

print_lti_pays(start_date=datetime.date(year=2021, month=9, day=30),
               end_date=datetime.date(year=2022, month=12, day=31))

if __name__ == "__main__":
    print(root_dir, pickle_dir, data_dir, sep='\n')
