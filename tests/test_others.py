#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from src.train.utils import flat_to_nested_config, merge_config_dicts


@pytest.mark.parametrize(
    'flat, nested',
    [
        (
            {
                'a': 'b',
                'c': 'd',
                'e': 'f',
            },
            {
                'a': 'b',
                'c': 'd',
                'e': 'f',
            },
        ),
        (
            {
                'a': 'b',
                'c.d': 'e',
                'c.f': 'g',
            },
            {
                'a': 'b',
                'c': {
                    'd': 'e',
                    'f': 'g',
                },
            },
        ),
        (
            {
                'a': 'b',
                'c.d': 'e',
                'c.f': 'g',
                'c.h.i': 'j',
                'c.h.k': 'l'
            },
            {
                'a': 'b',
                'c': {
                    'd': 'e',
                    'f': 'g',
                    'h': {
                        'i': 'j',
                        'k': 'l',
                    }
                },
            },
        ),
    ]
)
def test_flat_to_nest(flat: dict, nested: dict):
    assert flat_to_nested_config(flat) == nested


@pytest.mark.parametrize(
    'old_dict, new_dict, merged_dict',
    [
        (
            {
                'a': 'b',
                'c': 'd',
                'e': 'f',
            },
            {
                'a': 'b',
                'c': 'e',
            },
            {
                'a': 'b',
                'c': 'e',
                'e': 'f',
            }
        ),
        (
            {
                'a': 'b',
                'c': 'd',
                'e': {
                    'g': 'h',
                    'i': 'j',
                },
            },
            {
                'a': 'b',
                'c': 'e',
                'e': {
                    'g': 'i',
                }
            },
            {
                'a': 'b',
                'c': 'e',
                'e': {
                    'g': 'i',
                    'i': 'j',
                }
            }
        ),
    ]
)
def test_merge_configs(old_dict: dict, new_dict: dict, merged_dict: dict):
    assert merge_config_dicts(old_dict, new_dict) == merged_dict

