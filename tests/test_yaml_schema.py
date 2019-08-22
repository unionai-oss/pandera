"""Test the yaml loading capabilities in pandera.yaml_schema"""

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import pandera
from pandera import yaml_schema
import yaml


@pytest.fixture(scope='function')
def sample_df():
    """Multi-typed sample dataframe for the tests below"""
    return pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.5, 9.9],
        "str_col": ["z", "y", "x"],
        "bool_col": [True, True, False],
        "datetime_col": [
            pd.Timestamp("2015-02-01"),
            pd.Timestamp("2015-02-02"),
            pd.Timestamp("2015-02-03")
        ],
        "unexpected_col": [1, 1, 1]
    })


@pytest.fixture
def sample_df_schema():
    """YAML definition of a schema matching the dataframe of sample_df()"""
    return open(Path(__file__).parent / 'test_yaml_schema-sample_df_schema.yml')


def test_valid_yaml(sample_df_schema):
    """Simply check whether the test yaml is valid"""
    yaml.safe_load(sample_df_schema)


def test_schema_loadable(sample_df_schema):
    """Try to create a DataFrameSchema object out of the yaml definition"""
    yaml_schema.df_schema_from_yaml(sample_df_schema)


def test_schema_validates_unmodified(sample_df, sample_df_schema):
    """Unmodified the schema should be loadable and validate the sample df."""
    schema = yaml_schema.df_schema_from_yaml(sample_df_schema)
    schema.validate(sample_df)


def update_nested_dict(old_dict: Dict, change_path: str, new_value) -> Dict:
    """Update one value in a nested dict

    :param old_dict: The dict whose value should be changed
    :param change_path: The keys defining the key of the value to be changed.
        Dot separated string. E.g. pass 'a.b' to change the value 'd[a][b]' in
        the dictionary 'd'.
    :param new_value: The new value which should be assigned to the specified
        key.

    :return: A copy of the dictionary with the specified value being replaced.
    """
    keys = change_path.split('.')
    if not keys[0] in old_dict:
        raise KeyError("Key %s not found" % keys[0])
    new_dict = old_dict.copy()
    if len(keys) == 1:
        new_dict[keys[0]] = new_value
    elif len(keys) > 1:
        new_dict[keys[0]] = update_nested_dict(new_dict[keys[0]],
                                               '.'.join(keys[1:]), new_value)
    return new_dict


@pytest.mark.parametrize('old, keypath, value, new', [
    ({'a': 1}, 'a', 2, {'a': 2}),
    ({'a': {'b': 1}}, 'a.b', 2, {'a': {'b': 2}}),
    ({'a': {'b': 1}, 'b': 1}, 'a.b', 2, {'a': {'b': 2}, 'b': 1})
])
def test_update_nested_dict(old, keypath, value, new):
    assert update_nested_dict(old, keypath, value) == new


@pytest.mark.parametrize('old, keypath, value', [
    ({}, 'a', 2),
    ({}, 'a.b', 2),
    ({'a': 1}, 'b', 2),
    ({'a': {'b': 1}}, 'a.c', 2)
])
def test_update_nested_dict_key_error(old, keypath, value):
    with pytest.raises(KeyError):
        update_nested_dict(old, keypath, value)


@pytest.mark.parametrize("invalid_schema, error_match",
                         [("table:\n - x", "Expected key 'dataframe' missing"),
                          ("dataframe:\n checks:\n  - some_check",
                           "Invalid definition of dataframe check"),
                          ("dataframe:\n columns:\n  x: []",
                           "lacks a data type definition"),
                          ("dataframe:\n columns:\n  x:\n   dtype: thing",
                           "Unknown data type")])
def test_invalid_schema(invalid_schema, error_match):
    """Pass an invalid yaml file and see if a SchemaDefinitionError is raised.
    """
    with pytest.raises(pandera.SchemaDefinitionError, match=error_match):
        yaml_schema.df_schema_from_yaml(invalid_schema)


@pytest.mark.parametrize("change_path, new_value",
                         [('dataframe.strict', True)])
def test_schema_modifications(sample_df, sample_df_schema, change_path,
                              new_value):
    """Modify the yaml to find out if the individual changes are recognized.

    Each modification should lead to a validation failure.
    """
    schema_dict = yaml.safe_load(sample_df_schema)
    new_yaml = yaml.dump(update_nested_dict(schema_dict, change_path,
                                            new_value))

    schema = yaml_schema.df_schema_from_yaml(new_yaml)
    with pytest.raises(pandera.SchemaError):
        schema.validate(sample_df)
