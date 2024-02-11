import logging

from omegaconf import OmegaConf

import utils.path_utils as pu

from accounting.Account import Account, AccountStats, PREDICTED_BALANCE
from accounting.account_list import AccountsList

logging.basicConfig(level=logging.INFO)

accounts = list()
desjardins_accounts = list()

# TODO: make an Account factory

# Dynamically load Account objects int the accounting module from the corresponding directories
accounts_directories = [file for file in pu.accounts_dir.glob('*/')]
accounts_directories.sort()
for account_directory in accounts_directories:
    fname = account_directory.stem
    config_path =account_directory.joinpath(f'config.yaml')
    if config_path.exists():
        conf = OmegaConf.load(config_path.resolve())
        conf['account_dir'] = str(account_directory)
        globals()[fname] = Account(conf=conf)
        accounts.append(globals()[fname])
        if 'desjardins' in fname.lower():
            desjardins_accounts.append(globals()[fname])

accounts: AccountsList = AccountsList(accounts=accounts)
desjardins_accounts: AccountsList = AccountsList(accounts=desjardins_accounts)


# def create_new_account() -> None:
