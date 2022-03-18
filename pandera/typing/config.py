"""Class-based schema model API configuration."""

from typing import Any, Dict, List, Optional, Union

from ..schemas import PandasDtypeInputTypes
from .formats import Format


class BaseConfig:  # pylint:disable=R0903
    """Define DataFrameSchema-wide options.

    *new in 0.5.0*
    """

    #: datatype of the dataframe. This overrides the data types specified in
    #: any of the fields.
    dtype: Optional[PandasDtypeInputTypes] = None

    name: Optional[str] = None  #: name of schema
    title: Optional[str] = None  #: human-readable label for schema
    description: Optional[str] = None  #: arbitrary textual description
    coerce: bool = False  #: coerce types of all schema components

    #: make sure certain column combinations are unique
    unique: Optional[Union[str, List[str]]] = None

    #: make sure all specified columns are in the validated dataframe -
    #: if ``"filter"``, removes columns not specified in the schema
    strict: Union[bool, str] = False

    ordered: bool = False  #: validate columns order
    multiindex_name: Optional[str] = None  #: name of multiindex

    #: coerce types of all MultiIndex components
    multiindex_coerce: bool = False

    #: make sure all specified columns are in validated MultiIndex -
    #: if ``"filter"``, removes indexes not specified in the schema
    multiindex_strict: bool = False

    #: validate MultiIndex in order
    multiindex_ordered: bool = True

    #: make sure dataframe column names are unique
    unique_column_names: bool = False

    #: data format before validation. This option only applies to
    #: schemas used in the context of the pandera type constructor
    #: ``pa.typing.DataFrame[Schema](data)``. If None, assumes a data structure
    #: compatible with the ``pandas.DataFrame`` constructor.
    from_format: Optional[Format] = None

    #: a dictionary keyword arguments to pass into the reader function that
    #: converts the object of type ``from_format`` to a pandera-validate-able
    #: data structure. The reader function is implemented in the pandera.typing
    #: generic types via the ``from_format`` and ``to_format`` methods.
    from_format_kwargs: Optional[Dict[str, Any]] = None

    #: data format to serialize into after validation. This option only applies
    #: to  schemas used in the context of the pandera type constructor
    #: ``pa.typing.DataFrame[Schema](data)``. If None, returns a dataframe.
    to_format: Optional[Format] = None

    #: a dictionary keyword arguments to pass into the writer function that
    #: converts the pandera-validate-able object to type ``to_format``.
    #: The writer function is implemented in the pandera.typing
    #: generic types via the ``from_format`` and ``to_format`` methods.
    to_format_kwargs: Optional[Dict[str, Any]] = None
