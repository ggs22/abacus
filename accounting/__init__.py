import logging
import shutil
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import utils.path_utils as pu

from accounting.Account import Account, AccountStats, PREDICTED_BALANCE
from accounting.account_list import AccountsList

logging.basicConfig(level=logging.INFO)

accounts = list()


class AccountFactory:
    def __init__(self):
        self.accounts_dir: Path = pu.accounts_dir

    def load_accounts(self) -> AccountsList:
        # Dynamically load Account objects int the accounting module from the corresponding directories
        accounts_directories = [file for file in self.accounts_dir.glob('*/')]
        accounts_directories.sort()
        cmap_index = 0
        for account_directory in accounts_directories:
            fname = account_directory.stem
            config_path =account_directory.joinpath(f'config.yaml')
            if config_path.exists():
                conf = OmegaConf.load(config_path.resolve())
                conf['account_dir'] = str(account_directory)
                globals()[fname] = Account(conf=conf)
                globals()[fname].color = plt.cm.get_cmap('tab20')(cmap_index)
                cmap_index += 1
                accounts.append(globals()[fname])
        return AccountsList(accounts=accounts)

    def create_new_account(self) -> List[Account]:
        template_dir = self.accounts_dir.joinpath('_template_account')
        account_name = input("Enter the name of the new account:\n")
        dest_dir = self.accounts_dir.joinpath(account_name.lower())
        shutil.copytree(template_dir, dest_dir)
        shutil.move(dest_dir.joinpath('_config.yaml'), dest_dir.joinpath('config.yaml'))
        config = OmegaConf.load(dest_dir.joinpath('config.yaml'))
        config['name'] = account_name
        OmegaConf.save(config, dest_dir.joinpath('config.yaml'))
        return self.load_accounts()

    @staticmethod
    def add_account_property(accounts: AccountsList) -> None:

        property_name = input("Enter the property name:\n")

        for account in accounts:
            value = input(f"{account.name} - Enter the property value (or Enter for default 'None' value):\n")
            if not value:
                value = None
            account.conf[property_name] = value
            OmegaConf.save(account.conf, str(Path(account.account_dir).joinpath('config.yaml')))


