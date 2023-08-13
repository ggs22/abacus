from typing import List

from omegaconf import OmegaConf
from tqdm import tqdm

import utils.path_utils as pu

from accounting.Account import Account

accounts: List[Account] = list()
desjardins_accounts: List[Account] = list()

# Dynamically load NewRefactoredAccount objects int the accounting module from the corresponding yaml files
config_files = [file for file in pu.accounts_dir.glob('*.yaml')]
for yaml_file in tqdm(config_files, desc="Loading accounts..."):
    fname = yaml_file.stem
    conf = OmegaConf.load(yaml_file)
    globals()[fname] = Account(conf=conf)
    accounts.append(globals()[fname])
    if 'desjardins' in fname.lower():
        desjardins_accounts.append(globals()[fname])
