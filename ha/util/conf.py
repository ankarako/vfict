from typing import Dict, Any, List
from easydict import EasyDict
import yaml
import os
import random
import numpy as np
import torch

__all__ = ['read_conf', 'seed_all', 'get_device']

def make_easy(mapping: Dict[str, Any]) -> EasyDict:
    """
    Utility function for recursively creating an EasyDict object
    out of normal dict.

    :param mapping The dict object to parse.
    :return An EasyDict object from the specified dict
    """
    for key, value in mapping.items():
        if isinstance(value, dict):
            value = make_easy(value)
        mapping[key] = value
    return EasyDict(mapping)

def make_dict(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function for recursively creating a dictionary
    object.
    """
    if isinstance(mapping, EasyDict):
        mapping = dict(mapping)
    return {k: make_dict(v) if isinstance(v, EasyDict) else v for k, v in mapping.items()}


def read_conf(filepath: str) -> EasyDict:
    """
    Read the specified .yaml configuration file.

    :param filepath The path to the .yaml file to read.
    :return An EasyDict object with the specified file's contents
    """
    if not os.path.exists(filepath):
        raise ValueError(f"The specified configuration file does not exist: {filepath}.")
    
    with open(filepath, 'r') as infd:
        data = yaml.safe_load(infd)
    data = make_easy(data)
    return data

def save_conf(filepath: str, **kwargs) -> None:
    """
    Save the specified keyword arguments to a .yaml file

    :param filepath The path to save the file
    :param kwargs The dictionary to save basically.
    """
    conf = make_dict(dict(**kwargs))
    with open(filepath, 'w') as outfd:
        yaml.safe_dump(conf, outfd)


def seed_all(seed: int) -> None:
    """
    Utility function for seeding all the libraries
    with the same random seed.

    :param seed The seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_deterministic(deterministic: bool=True) -> None:
    # torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic


def get_device(device_list: List[int]) -> torch.device:
    if torch.cuda.is_available():
        if len(device_list) > 1:
            raise NotImplementedError(f"More than two cuda device not supported yet")
        return torch.device(f"cuda:{device_list[0]}")
    else:
        return torch.device("cpu")
