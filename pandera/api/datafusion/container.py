"""Core DataFusion DataFrame container specification."""

from __future__ import annotations

import warnings

import datafusion

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.datafusion.utils import get_validation_depth, resolve_dtype
from pandera.backends.datafusion.register import register_datafusion_backends
from pandera.config import config_context, get_config_context


class DataFrameSchema(_DataFrameSchema[datafusion.DataFrame]):
    """A lightweight DataFusion DataFrame validator."""

    def _validate_attributes(self):
        super()._validate_attributes()

        if self.report_duplicates != "all":
            warnings.warn(
                "Setting report_duplicates to 'exclude_first' or "
                "'exclude_last' will have no effect on validation. With the "
                "DataFusion backend, all duplicate values will be reported."
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
        """Validate a DataFusion DataFrame against the schema.

        A ``datafusion.DataFrame`` is a lazy query plan. By default only
        schema-level checks (column presence, dtypes) run; set the
        validation depth to ``SCHEMA_AND_DATA`` to run data-level checks as
        well. See
        :ref:`DataFusion <datafusion>` for details.

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

        :raises SchemaError: when ``check_obj`` violates built-in or custom
            checks.

        :example:

        >>> import pyarrow
        >>> from datafusion import SessionContext
        >>> import pandera.datafusion as pa
        >>>
        >>> ctx = SessionContext()
        >>> df = ctx.from_arrow(pyarrow.table({"probability": [0.1, 0.4]}))
        >>> schema = pa.DataFrameSchema({
        ...     "probability": pa.Column(float, pa.Check.le(1.0)),
        ... })
        >>> schema.validate(df).to_arrow_table()
        pyarrow.Table
        probability: double
        ----
        probability: [[0.1,0.4]]
        """
        if not get_config_context().validation_enabled:
            return check_obj

        with config_context(validation_depth=get_validation_depth(check_obj)):
            output = self.get_backend(check_obj).validate(
                check_obj=check_obj,
                schema=self,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )

        return output

    @_DataFrameSchema.dtype.setter  # type: ignore[attr-defined]
    def dtype(self, value) -> None:
        """Set the dtype property."""
        self._dtype = resolve_dtype(value)

    def strategy(self, *, size: int | None = None, n_regex_columns: int = 1):
        """Data synthesis is not supported for DataFusion schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with DataFusion schemas."
        )

    def example(self, size: int | None = None, n_regex_columns: int = 1):
        """Data synthesis is not supported for DataFusion schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with DataFusion schemas."
        )
