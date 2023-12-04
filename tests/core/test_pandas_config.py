"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method,redefined-outer-name


import pandas as pd
import pytest

import pandera as pa
from pandera import DataFrameModel, DataFrameSchema, SeriesSchema
from pandera.config import CONFIG, ValidationDepth


@pytest.fixture()
def disable_validation():
    """Fixture to disable validation and clean up after the test is finished"""
    CONFIG.validation_enabled = False
    yield "resource"
    CONFIG.validation_enabled = True


class TestPandasDataFrameConfig:
    """Class to test all the different configs types"""

    sample_data = pd.DataFrame(
        (("Bread", 9), ("Cutter", 15)), columns=["product", "price_val"]
    )
    # pylint: disable=unused-argument
    def test_disable_validation(self, disable_validation):
        """This function validates that a none object is loaded if validation is disabled"""

        pandera_schema = DataFrameSchema(
            {
                "product": pa.Column(
                    str, pa.Check(lambda s: s.str.startswith("B"))
                ),
                "price_val": pa.Column(int),
            }
        )

        class TestSchema(DataFrameModel):
            """Test Schema class"""

            product: str = pa.Field(str_startswith="B")
            price_val: int = pa.Field()

        expected = {
            "validation_enabled": False,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
        }

        assert CONFIG.dict() == expected
        assert pandera_schema.validate(self.sample_data) is self.sample_data
        assert TestSchema.validate(self.sample_data) is self.sample_data


class TestPandasSeriesConfig:
    """Class to test all the different configs types"""

    sample_data = pd.Series([1, 1, 2, 2, 3, 3])
    # pylint: disable=unused-argument
    def test_disable_validation(self, disable_validation):
        """This function validates that a none object is loaded if validation is disabled"""
        expected = {
            "validation_enabled": False,
            "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
        }
        pandera_schema = SeriesSchema(
            int, pa.Check(lambda s: s.value_counts() == 2, element_wise=False)
        )
        assert CONFIG.dict() == expected
        assert pandera_schema.validate(self.sample_data) is self.sample_data
