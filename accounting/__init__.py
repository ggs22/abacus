import logging

from omegaconf import OmegaConf

import utils.path_utils as pu

from accounting.Account import Account
from accounting.account_list import AccountsList

logging.basicConfig(level=logging.INFO)

accounts = list()
desjardins_accounts = list()

# Dynamically load NewRefactoredAccount objects int the accounting module from the corresponding yaml files
config_files = [file for file in pu.accounts_dir.glob('*.yaml')]
config_files.sort()
for yaml_file in config_files:
    fname = yaml_file.stem
    conf = OmegaConf.load(yaml_file)
    globals()[fname] = Account(conf=conf)
    accounts.append(globals()[fname])
    if 'desjardins' in fname.lower():
        desjardins_accounts.append(globals()[fname])

accounts: AccountsList = AccountsList(accounts=accounts)
desjardins_accounts: AccountsList = AccountsList(accounts=desjardins_accounts)
