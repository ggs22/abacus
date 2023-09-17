from typing import List

from omegaconf import OmegaConf
from tqdm import tqdm

import utils.path_utils as pu

from accounting.Account import Account
from accounting.account_list import AccountsList
from accounting.prediction_strategies import get_balance_prediction

accounts = list()
desjardins_accounts = list()

# Dynamically load NewRefactoredAccount objects int the accounting module from the corresponding yaml files
config_files = [file for file in pu.accounts_dir.glob('*.yaml')]
for yaml_file in tqdm(config_files, desc="Loading accounts..."):
    fname = yaml_file.stem
    conf = OmegaConf.load(yaml_file)
    globals()[fname] = Account(conf=conf, predict=get_balance_prediction)
    accounts.append(globals()[fname])
    if 'desjardins' in fname.lower():
        desjardins_accounts.append(globals()[fname])

accounts: AccountsList = AccountsList(accounts=accounts)
desjardins_accounts: AccountsList = AccountsList(accounts=desjardins_accounts)
