#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy


def flat_to_nested_config(config: dict) -> dict:
    """Build a nested configuration dictionnary from a flatten configuration.
    A key is representing a nested dictionnary if it has a '.' in its name.
    For example:
        `{'a': 'b', 'c.d': 'e', 'c.f': 'g'}`
    Becomes:
        `{'a': 'b', 'c': {'d': 'e', 'f': 'g'}}`
    """
    nested = dict()
    for k, v in config.items():
        # It first creates and dive into the sub dictionnaries
        current_dict = nested
        sub_keys = k.split('.')
        for sub_k in sub_keys[:-1]:
            if sub_k not in current_dict:
                # The subdict is not created yet
                current_dict[sub_k] = dict()
            current_dict = current_dict[sub_k]

        # Apply the last key and the value associated with
        current_dict[sub_keys[-1]] = v

    return nested


def merge_config_dicts(old_dict: dict, new_dict: dict) -> dict:
    """Merge both dicts into one main dictionnary.
    In case there is a collision of data, the data from the `new_dict` is kept.

    Configurations can be nested.
    """
    merged = deepcopy(old_dict)

    for k, v in new_dict.items():
        if k not in merged:
            merged[k] = v
            continue

        if type(v) is dict:
            merged[k] = merge_config_dicts(merged[k], v)
        else:
            merged[k] = v  # Overwrite

    return merged

