from pathlib import Path
from typing import Callable


def ensure_dir_exists(path: Callable) -> Path:
    path: Path = path()
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_path_exists(path: Callable) -> Path:
    path: Path = path()
    if not path.exists():
        raise FileNotFoundError(f"The file {path} wasn't found!")
    else:
        return path


@ensure_dir_exists
def root_dir() -> Path:
    rdir = Path(__file__).parent.parent
    return rdir


@ensure_dir_exists
def accounting_dir() -> Path:
    return root_dir.joinpath('accounting')


@ensure_dir_exists
def accounting_data_dir() -> Path:
    return accounting_dir.joinpath('data')


@ensure_dir_exists
def accounts_dir() -> Path:
    return accounting_dir.joinpath('accounts')


@ensure_dir_exists
def pickle_objects_dir() -> Path:
    return root_dir.joinpath('pickle_objects')


@validate_path_exists
def common_assignations_path() -> Path:
    return accounting_data_dir.joinpath('assignation.json')