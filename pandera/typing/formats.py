"""Serialization formats for dataframes."""

from enum import Enum
from typing import Literal, Union


class Formats(Enum):
    """Data container serialization formats.

    The values of this enum specify the valid values taken by the ``to_format``
    and ``from_format`` attributes in
    :py:class:`~pandera.typing.config.BaseConfig` when specifying a
    :py:class:`~pandera.api.pandas.model.DataFrameModel`.
    """

    #: comma-separated values file
    csv = "csv"

    #: python dictionary
    dict = "dict"

    #: json file
    json = "json"

    #: feather file format. See
    #: `here <https://arrow.apache.org/docs/python/feather.html>`__ for more
    #: details
    feather = "feather"

    #: parquet file format. See `here <https://parquet.apache.org/>`__ for more
    #: details
    parquet = "parquet"

    #: python pickle file format
    pickle = "pickle"

    #: python json_normalize
    json_normalize = "json_normalize"


Format = Union[
    Literal[Formats.csv],
    Literal[Formats.dict],
    Literal[Formats.json],
    Literal[Formats.feather],
    Literal[Formats.parquet],
    Literal[Formats.pickle],
    Literal[Formats.json_normalize],
]
