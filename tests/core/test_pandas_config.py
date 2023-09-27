"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method

import pyspark.sql.types as T

from pandera.config import CONFIG, ValidationDepth
from pandera.pyspark import (
    Check,
    DataFrameSchema,
    Column,
    DataFrameModel,
    Field,
)
from tests.pyspark.conftest import spark_df


class TestPanderaConfig:
    ...
    