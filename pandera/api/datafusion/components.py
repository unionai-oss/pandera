"""Core DataFusion schema component specifications."""

from __future__ import annotations

from typing import Any

import datafusion
import narwhals.stable.v1 as nw

from pandera.api.base.types import CheckList
from pandera.api.datafusion.types import DataFusionDtypeInputTypes
from pandera.api.datafusion.utils import get_validation_depth
from pandera.api.pyarrow.components import Column as _PyArrowColumn
from pandera.backends.datafusion.register import register_datafusion_backends
from pandera.config import config_context, get_config_context


class Column(_PyArrowColumn):
    """Validate types and properties of DataFusion DataFrame columns.

    DataFusion dtypes are pyarrow end-to-end, so this component resolves
    dtypes exactly like :class:`pandera.api.pyarrow.components.Column` and
    only differs in which native frame type it dispatches on.
    """

    def __init__(
        self,
        dtype: DataFusionDtypeInputTypes | None = None,
        checks: CheckList | None = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: str | None = None,
        regex: bool = False,
        title: str | None = None,
        description: str | None = None,
        default: Any | None = None,
        metadata: dict | None = None,
        drop_invalid_rows: bool = False,
        **column_kwargs,
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. Accepts native ``pyarrow``
            types (e.g. ``pyarrow.int64()``), narwhals dtypes, the pandera
            abstract datatypes, supported python builtins (``int``, ``float``,
            ``str``, ``bool``) and their string aliases.
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in the dataframe to validate. Names in the
            format '^{regex_pattern}$' are treated as regular expressions.
            During validation, this schema will be applied to any columns
            matching this pattern.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the column.
        :param default: The default value for missing values in the column.
        :param metadata: An optional key value data.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pyarrow
        >>> from datafusion import SessionContext
        >>> import pandera.datafusion as pa
        >>>
        >>> ctx = SessionContext()
        >>> df = ctx.from_arrow(pyarrow.table({"column": ["foo", "bar"]}))
        >>> schema = pa.DataFrameSchema({"column": pa.Column(str)})
        >>> schema.validate(df).to_arrow_table()
        pyarrow.Table
        column: string
        ----
        column: [["foo","bar"]]
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            unique=unique,
            coerce=coerce,
            required=required,
            name=name,
            regex=regex,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
            **column_kwargs,
        )

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        register_datafusion_backends()

    def validate(
        self,
        check_obj: datafusion.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> datafusion.DataFrame:
        """Validate a column of a DataFusion DataFrame.

        Follows :meth:`DataFrameSchema.validate
        <pandera.api.datafusion.container.DataFrameSchema.validate>`: only
        schema-level checks run unless the validation depth is
        ``SCHEMA_AND_DATA``.

        :param check_obj: the ``datafusion.DataFrame`` to be validated.
        :param head: validate the first n rows.
        :param tail: not supported; DataFusion plans have no ``tail``.
        :param sample: not supported by the narwhals backend.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates the dataframe against all
            validation checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: has no effect; DataFusion DataFrames are immutable.
        :returns: validated ``datafusion.DataFrame``
        """
        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=get_validation_depth(check_obj)):
            output = self.get_backend(check_obj).validate(
                check_obj,
                schema=self,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

        return nw.to_native(output, pass_through=True)

    def strategy(self, *, size=None):
        """Data synthesis is not supported for DataFusion schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with DataFusion schemas."
        )

    def strategy_component(self):
        """Data synthesis is not supported for DataFusion schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with DataFusion schemas."
        )

    def example(self, size=None):
        """Data synthesis is not supported for DataFusion schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with DataFusion schemas."
        )
