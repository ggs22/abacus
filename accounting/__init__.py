import os
from typing import List

import utils.path_utils as pu
import accounting.Account
from accounting.CIBC import CIBC
from accounting.CapitalOne import CapitalOne
from accounting.DesjardinsMC import DesjardinsMC
from accounting.DesjardinsOP import DesjardinsOP
from accounting.DesjardinsPR import DesjardinsPR
from accounting.Paypal import PayPal
from accounting.VisaPP import VisaPP
from accounting.Account import AccountNames, AccountStatus, AccountType, AccountMetadata, NewRefactoredAccount
from utils.utils import data_dir, pickle_dir
from omegaconf import OmegaConf

# Dynamically load Account2 objects int the accounting module from the corresponding yaml files
for yaml_file in pu.accounts_dir.glob('*.yaml'):
    fname = yaml_file.stem
    conf = OmegaConf.load(yaml_file)
    globals()[fname] = NewRefactoredAccount(conf=conf)

# """
# Create account objects
# """
# # ==============
# # Desjardins OP
# # ==============
# names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal', 'deposit',
#          'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']
#
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'desjardins_op.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0,
#                            name=AccountNames.DESJARDINS_OP,
#                            columns_names=names,
#                            assignation_file_path=os.path.join(data_dir, 'assignations_desjardins_op.json'),
#                            type=AccountType.OPERATIONS,
#                            status=AccountStatus.OPEN)
# desjardins_op = DesjardinsOP(lmetadata=metadata)
#
# # ==============
# # Desjardins MC
# # ==============
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'desjardins_mc.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0.072,
#                            name=AccountNames.DESJARDINS_MC,
#                            columns_names=names,
#                            assignation_file_path=os.path.join(data_dir, 'assignations_desjardins_mc.json'),
#                            type=AccountType.CREDIT,
#                            status=AccountStatus.OPEN)
# desjardins_mc = DesjardinsMC(lmetadata=metadata)
#
# # ==============
# # Desjardins PR
# # ==============
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'desjardins_pr.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0.072,
#                            name=AccountNames.DESJARDINS_PR,
#                            columns_names=names,
#                            assignation_file_path=os.path.join(data_dir, 'assignations_desjardins_pr.json'),
#                            type=AccountType.CREDIT,
#                            status=AccountStatus.OPEN)
# desjardins_pr = DesjardinsPR(lmetadata=metadata)
#
# # ==============
# # Desjardins PrePaid VISA
# # ==============
# names = ['acc_name', 'unknown1', 'unknown2',  'date', 'transaction_num', 'description', 'unknown6', 'unknown7',
#          'unknown8', 'unknown9', 'unknown10', 'payment/credit',  'credit', 'unknown13']
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_ppcard_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'desjardins_ppcard.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0,
#                            name=AccountNames.VISA_PP,
#                            columns_names=names,
#                            assignation_file_path='',
#                            type=AccountType.PREPAID,
#                            status=AccountStatus.OPEN)
# visapp = VisaPP(lmetadata=metadata)
#
# # ==============
# # Capital One
# # ==============
# names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'capital_one_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'capital_one.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0,
#                            name=AccountNames.CAPITAL_ONE,
#                            columns_names=names,
#                            assignation_file_path='',
#                            type=AccountType.CREDIT,
#                            status=AccountStatus.CLOSED)
# capital_one = CapitalOne(lmetadata=metadata)
#
# # ==============
# # CIBC
# # ==============
# names = ['date', 'description', 'debit', 'credit', 'card_num']
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'cibc_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'cibc.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0.1924,
#                            name=AccountNames.CIBC,
#                            columns_names=names,
#                            assignation_file_path=os.path.join(data_dir, 'assignations_cibc.json'),
#                            type=AccountType.CREDIT,
#                            status=AccountStatus.OPEN)
# cibc = CIBC(lmetadata=metadata)
#
# # ==============
# # PayPal
# # ==============
# names = ['Date', 'Time', 'TimeZone', 'Name', 'description', 'Status', 'Currency', 'Amount', 'Receipt ID', 'Balance']
# metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'paypal_csv_files'),
#                            serialized_object_path=os.path.join(pickle_dir, 'paypal.pkl'),
#                            planned_transactions_path=os.path.join(data_dir,
#                                                                   'desjardins_planned_transactions.csv'),
#                            interest_rate=0,
#                            name=AccountNames.PAYPAL,
#                            columns_names=names,
#                            assignation_file_path='',
#                            type=AccountType.PREPAID,
#                            status=AccountStatus.CLOSED)
# paypal = PayPal(lmetadata=metadata)
#
# # Create the Accounts object allowing to integrate the data of all the accounting
# accounts = accounting.Account.Accounts(l_accounts_list=[desjardins_op,
#                                                         desjardins_mc,
#                                                         desjardins_pr,
#                                                         visapp,
#                                                         capital_one,
#                                                         cibc,
#                                                         paypal])
#
# desjardins_accounts = accounting.Account.Accounts(l_accounts_list=[desjardins_op,
#                                                                    desjardins_mc,
#                                                                    desjardins_pr])
