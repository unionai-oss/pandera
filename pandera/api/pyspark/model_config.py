"""Class-based dataframe model API configuration for pyspark."""

from typing import Any, Callable, Dict, List, Optional, Union

from pandera.api.base.model_config import BaseModelConfig
from pandera.api.base.types import StrictType
from pandera.api.pyspark.types import PySparkDtypeInputTypes
from pandera.typing.formats import Format


class BaseConfig(BaseModelConfig):  # pylint:disable=R0903
    """Define DataFrameSchema-wide options.

    *new in 0.16.0*
    """

    #: datatype of the dataframe. This overrides the data types specified in
    #: any of the fields.
    dtype: Optional[PySparkDtypeInputTypes] = None

    name: Optional[str] = None  #: name of schema
    title: Optional[str] = None  #: human-readable label for schema
    description: Optional[str] = None  #: arbitrary textual description
    coerce: bool = False  #: coerce types of all schema components

    #: make sure certain column combinations are unique
    unique: Optional[Union[str, List[str]]] = None

    #: make sure all specified columns are in the validated dataframe -
    #: if ``"filter"``, removes columns not specified in the schema
    strict: StrictType = False

    ordered: bool = False  #: validate columns order

    #: make sure dataframe column names are unique
    unique_column_names: bool = False

    #: data format before validation. This option only applies to
    #: schemas used in the context of the pandera type constructor
    #: ``pa.typing.DataFrame[Schema](data)``. If None, assumes a data structure
    #: compatible with the ``pyspark.sql.DataFrame`` constructor.
    from_format: Optional[Union[Format, Callable]] = None

    #: a dictionary keyword arguments to pass into the reader function that
    #: converts the object of type ``from_format`` to a pandera-validate-able
    #: data structure. The reader function is implemented in the pandera.typing
    #: generic types via the ``from_format`` and ``to_format`` methods.
    from_format_kwargs: Optional[Dict[str, Any]] = None

    #: data format to serialize into after validation. This option only applies
    #: to  schemas used in the context of the pandera type constructor
    #: ``pa.typing.DataFrame[Schema](data)``. If None, returns a dataframe.
    to_format: Optional[Union[Format, Callable]] = None

    #: Buffer to be provided when to_format is a custom callable. See docs for
    #: example of how to implement an example of a to format function.
    to_format_buffer: Optional[Union[str, Callable]] = None

    #: a dictionary keyword arguments to pass into the writer function that
    #: converts the pandera-validate-able object to type ``to_format``.
    #: The writer function is implemented in the pandera.typing
    #: generic types via the ``from_format`` and ``to_format`` methods.
    to_format_kwargs: Optional[Dict[str, Any]] = None

    #: a dictionary object to store key-value data at schema level
    metadata: Optional[dict] = None
