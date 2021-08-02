#%%
import os
import Accounts
from Accounts import DesjardinsOP, DesjardinsMC, DesjardinsPR, VisaPP, CapitalOne

print(os.path.abspath(__file__))

os.chdir('/home/ggsanchez/repos/abacus')
base_dir = '/home/ggsanchez/repos/abacus'
data_dir = os.path.join(base_dir, 'data')
pickle_dir = os.path.join(base_dir, 'pickle_objects')

# ==============
# Desjardins OP
# ==============
names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal', 'deposit',
         'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

metadata = Accounts.AccountMetadata(raw_files_path=os.path.join(data_dir, 'desjardins_csv_files'),
                                    serialized_object_path=os.path.join(pickle_dir, 'desjardins_op.pkl'),
                                    planned_transactions_path=os.path.join(data_dir, 'desjardins_planned_transactions.csv'),
                                    interest_rate=0,
                                    name=Accounts.AccountNames.DESJARDINSOP,
                                    columns_names=names,
                                    type=Accounts.AccountType.OPERATIONS,
                                    status=Accounts.AccountStatus.OPEN)
desjardins_op = DesjardinsOP(metadata=metadata)
# desjardins_op.update_from_csv()
desjardins_op.get_data_by_date()
pop = desjardins_op.get_predicted_balance()

# ==============
# Desjardins MC
# ==============
metadata = Accounts.AccountMetadata(raw_files_path=os.path.join(data_dir, 'desjardins_csv_files'),
                                    serialized_object_path=os.path.join(pickle_dir, 'desjardins_mc.pkl'),
                                    planned_transactions_path=os.path.join(data_dir, 'desjardins_planned_transactions.csv'),
                                    interest_rate=0.045,
                                    name=Accounts.AccountNames.DESJARDINSMC,
                                    columns_names=names,
                                    type=Accounts.AccountType.CREDIT,
                                    status=Accounts.AccountStatus.OPEN)
desjardins_mc = DesjardinsMC(metadata=metadata)
desjardins_mc.get_data_by_date()
# desjardins_mc.update_from_csv()
pmc = desjardins_mc.get_predicted_balance()

# ==============
# Desjardins PR
# ==============
metadata = Accounts.AccountMetadata(raw_files_path=os.path.join(data_dir, 'desjardins_csv_files'),
                                    serialized_object_path=os.path.join(pickle_dir, 'desjardins_pr.pkl'),
                                    planned_transactions_path=os.path.join(data_dir, 'desjardins_planned_transactions.pkl'),
                                    interest_rate=0.045,
                                    name=Accounts.AccountNames.DESJARDINSPR,
                                    columns_names=names,
                                    type=Accounts.AccountType.CREDIT,
                                    status=Accounts.AccountStatus.OPEN)
desjardins_pr = DesjardinsPR(metadata=metadata)
desjardins_pr.get_data_by_date()
# desjardins_pr.update_from_csv()
ppr = desjardins_pr.get_predicted_balance()

# ==============
# Desjardins PrePaid VISA
# ==============
names = ['date', 'transaction_num', 'description', 'credit/payment']
metadata = Accounts.AccountMetadata(raw_files_path=os.path.join(data_dir, 'desjardins_ppcard_pdf_files'),
                                    serialized_object_path=os.path.join(pickle_dir, 'desjardins_ppcard.pkl'),
                                    planned_transactions_path=os.path.join(data_dir,'desjardins_planned_transactions.pkl'),
                                    interest_rate=0,
                                    name=Accounts.AccountNames.VISAPP,
                                    columns_names=names,
                                    type=Accounts.AccountType.PREPAID,
                                    status=Accounts.AccountStatus.OPEN)
visapp = VisaPP(metadata=metadata)
visapp.get_data_by_date()
# visapp.update_from_csv()
ppr = visapp.get_predicted_balance()

# ==============
# Capital One
# ==============
names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']
metadata = Accounts.AccountMetadata(raw_files_path=os.path.join(data_dir, 'capital_one_csv_files'),
                                    serialized_object_path=os.path.join(pickle_dir, 'capital_one.pkl'),
                                    planned_transactions_path=os.path.join(data_dir, 'desjardins_planned_transactions.pkl'),
                                    interest_rate=0,
                                    name=Accounts.AccountNames.CAPITALONE,
                                    columns_names=names,
                                    type=Accounts.AccountType.CREDIT,
                                    status=Accounts.AccountStatus.CLOSED)
capital_one = CapitalOne(metadata=metadata)
visapp.get_data_by_date()
# visapp.update_from_csv()
ppr = visapp.get_predicted_balance()
