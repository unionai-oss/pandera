"""Core PyArrow table container specification."""

from __future__ import annotations

import warnings

import pyarrow as pa

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.api.pyarrow.utils import get_validation_depth, resolve_dtype
from pandera.backends.pyarrow.register import register_pyarrow_backends
from pandera.config import config_context, get_config_context


class DataFrameSchema(_DataFrameSchema[pa.Table]):
    """A lightweight pyarrow Table validator."""

    def _validate_attributes(self):
        super()._validate_attributes()

        if self.unique_column_names:
            warnings.warn(
                "unique_column_names=True will have no effect on validation "
                "since pyarrow Tables do not support duplicate column names."
            )

        if self.report_duplicates != "all":
            warnings.warn(
                "Setting report_duplicates to 'exclude_first' or "
                "'exclude_last' will have no effect on validation. With the "
                "pyarrow backend, all duplicate values will be reported."
            )

    @staticmethod
    def register_default_backends(check_obj_cls: type):
        register_pyarrow_backends()

    def validate(
        self,
        check_obj: pa.Table,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pa.Table:
        """Validate a pyarrow Table against the schema.

        :param check_obj: the ``pyarrow.Table`` to be validated.
        :param head: validate the first n rows. Rows overlapping with ``tail``
            or ``sample`` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with ``head``
            or ``sample`` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with ``head`` or ``tail`` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates the table against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: has no effect; pyarrow Tables are immutable.
        :returns: validated ``pyarrow.Table``

        :raises SchemaError: when ``check_obj`` violates built-in or custom
            checks.

        :example:

        >>> import pyarrow
        >>> import pandera.pyarrow as pa
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "probability": pa.Column(float, pa.Check.le(1.0)),
        ... })
        >>> schema.validate(pyarrow.table({"probability": [0.1, 0.4]}))
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
        """Data synthesis is not supported for pyarrow schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with pyarrow schemas."
        )

    def example(self, size: int | None = None, n_regex_columns: int = 1):
        """Data synthesis is not supported for pyarrow schemas."""
        raise NotImplementedError(
            "Data synthesis is not supported with pyarrow schemas."
        )
