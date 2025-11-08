import logging
import shutil
from enum import StrEnum
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

    class AccountGroups(StrEnum):
        PERSONAL = "Personal"
        DESJARDINS = "Desjardins"
        NATIONALBANK = "NationalBank"
        WEALTHSIMPLE = "WealthSimple"
        ORDIAL = "Ordial"

    def __init__(self):
        self.accounts_dir: Path = pu.accounts_dir
        self.accounts = self._load_accounts()

    def get_account_groups(self):
        return  self.AccountGroups

    def get_account_from_group(self, group: AccountGroups) -> AccountsList:
        accounts_group = list()
        for acc in accounts:
            if group != self.AccountGroups.PERSONAL:
                if group.lower() in acc.name.lower():
                    accounts_group.append(acc)
            elif group == self.AccountGroups.PERSONAL:
                if not 'ordial' in acc.name.lower():
                    accounts_group.append(acc)
            else:
                raise ValueError(f"The parameter group muse belong to the following enum: {self.AccountGroups}")
        return AccountsList(accounts_group)

    def _load_accounts(self) -> AccountsList:
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

    def create_new_account(self) -> AccountsList:
        template_dir = self.accounts_dir.joinpath('_template_account')
        account_name = input("Enter the name of the new account:\n")
        dest_dir = self.accounts_dir.joinpath(account_name.lower())
        shutil.copytree(template_dir, dest_dir)
        shutil.move(dest_dir.joinpath('_config.yaml'), dest_dir.joinpath('config.yaml'))
        config = OmegaConf.load(dest_dir.joinpath('config.yaml'))
        config['name'] = account_name
        OmegaConf.save(config, dest_dir.joinpath('config.yaml'))
        return self._load_accounts()

    @staticmethod
    def add_account_property(accounts: AccountsList) -> None:

        property_name = input("Enter the property name:\n")

        for account in accounts:
            value = input(f"{account.name} - Enter the property value (or Enter for default 'None' value):\n")
            if not value:
                value = None
            account.conf[property_name] = value
            OmegaConf.save(account.conf, str(Path(account.account_dir).joinpath('config.yaml')))


