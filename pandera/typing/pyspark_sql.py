"""Pandera type annotations for Pyspark SQL."""

import functools
import json
from typing import Union, TypeVar, Any, get_args, Generic

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from pandera.engines import pyspark_engine, PYDANTIC_V2
from pandera.errors import SchemaInitError
from pandera.typing.common import DataFrameBase, _GenericAlias
from pandera.api.pyspark import DataFrameModel

try:
    import pyspark.sql as ps

    PYSPARK_SQL_INSTALLED = True
except ImportError:  # pragma: no cover
    PYSPARK_SQL_INSTALLED = False

T = TypeVar("T", bound=DataFrameModel)

if PYSPARK_SQL_INSTALLED:
    PysparkString = pyspark_engine.String
    PysparkInt = pyspark_engine.Int
    PysparkLongInt = pyspark_engine.BigInt
    PysparkShortInt = pyspark_engine.ShortInt
    PysparkByteInt = pyspark_engine.ByteInt
    PysparkDouble = pyspark_engine.Double
    PysparkFloat = pyspark_engine.Float
    PysparkDecimal = pyspark_engine.Decimal
    PysparkDate = pyspark_engine.Date
    PysparkTimestamp = pyspark_engine.Timestamp
    PysparkBinary = pyspark_engine.Binary

    PysparkDType = TypeVar(  # type: ignore
        "PysparkDType",
        bound=Union[
            PysparkString,  # type: ignore
            PysparkInt,  # type: ignore
            PysparkLongInt,  # type: ignore
            PysparkShortInt,  # type: ignore
            PysparkByteInt,  # type: ignore
            PysparkDouble,  # type: ignore
            PysparkFloat,  # type: ignore
            PysparkDecimal,  # type: ignore
            PysparkDate,  # type: ignore
            PysparkTimestamp,  # type: ignore
            PysparkBinary,  # type: ignore
        ],
    )

    if PYSPARK_SQL_INSTALLED:
        # pylint: disable=too-few-public-methods,arguments-renamed
        class DataFrame(DataFrameBase, ps.DataFrame, Generic[T]):
            """
            Representation of dask.dataframe.DataFrame, only used for type
            annotation.

            *new in 0.8.0*
            """

            def __class_getitem__(cls, item):
                """Define this to override's pyspark.pandas generic type."""
                return _GenericAlias(cls, item)  # pragma: no cover

            @classmethod
            def pydantic_validate(
                cls, obj: ps.DataFrame, schema_model: T
            ) -> ps.DataFrame:
                """
                Verify that the input can be converted into a pandas dataframe that
                meets all schema requirements.

                This is for pydantic V1 and V2.
                """
                try:
                    schema = schema_model.to_schema()
                except SchemaInitError as exc:
                    error_message = (
                        f"Cannot use {cls} as a pydantic type as its "
                        "DataFrameModel cannot be converted to a DataFrameSchema.\n"
                        f"Please revisit the model to address the following errors:"
                        f"\n{exc}"
                    )
                    raise ValueError(error_message) from exc

                validated_data = schema.validate(obj)

                if validated_data.pandera.errors:
                    errors = json.dumps(
                        dict(validated_data.pandera.errors), indent=4
                    )
                    raise ValueError(errors)

                return validated_data

            if PYDANTIC_V2:

                @classmethod
                def __get_pydantic_core_schema__(
                    cls, _source_type: Any, _handler: GetCoreSchemaHandler
                ) -> core_schema.CoreSchema:
                    schema_model = get_args(_source_type)[0]
                    return core_schema.no_info_plain_validator_function(
                        functools.partial(
                            cls.pydantic_validate,
                            schema_model=schema_model,
                        ),
                    )

            else:

                @classmethod
                def __get_validators__(cls):
                    yield cls._pydantic_validate

                @classmethod
                def _get_schema_model(cls, field):
                    if not field.sub_fields:
                        raise TypeError(
                            "Expected a typed pandera.typing.DataFrame,"
                            " e.g. DataFrame[Schema]"
                        )
                    schema_model = field.sub_fields[0].type_
                    return schema_model

                @classmethod
                def _pydantic_validate(cls, obj: Any, field) -> ps.DataFrame:
                    """
                    Verify that the input can be converted into a pandas dataframe that
                    meets all schema requirements.

                    This is for pydantic v1
                    """
                    schema_model = cls._get_schema_model(field)
                    return cls.pydantic_validate(obj, schema_model)
