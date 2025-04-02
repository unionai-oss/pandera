"""Core Ibis table container specification."""

from typing import Optional, Type

import ibis.expr.types as ir

from pandera.api.dataframe.container import DataFrameSchema as _DataFrameSchema
from pandera.backends.ibis.register import register_ibis_backends
from pandera.engines import ibis_engine


class DataFrameSchema(_DataFrameSchema[ir.Table]):
    """A lightweight Ibis table validator."""

    # pylint: disable=unused-argument
    @staticmethod
    def register_default_backends(check_obj_cls: Type):
        register_ibis_backends()

    def validate(
        self,
        check_obj: ir.Table,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> ir.Table:
        """Validate an Ibis table against the schema.

        :param ir.Table check_obj: the table to be validated.
        :param head: validate the first n rows. Rows overlapping with ``tail`` or
            ``sample`` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with ``head`` or
            ``sample`` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with ``head`` or ``tail`` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``ir.Table``

        :raises SchemaError: when ``check_obj`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the table.

        >>> import ibis
        >>> import pandas as pd
        >>> import pandera.ibis as pa
        >>>
        >>> df = pd.DataFrame({
        ...     "probability": [0.1, 0.4, 0.52, 0.23, 0.8, 0.76],
        ...     "category": ["dog", "dog", "cat", "duck", "dog", "dog"],
        ... })
        >>> t = ibis.memtable(df, name="t")
        >>>
        >>> schema_withchecks = pa.DataFrameSchema({
        ...     "probability": pa.Column(
        ...         float, pa.Check(lambda s: (s >= 0) & (s <= 1))),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": pa.Column(
        ...         str, [
        ...             pa.Check(lambda s: s.isin(["dog", "cat", "duck"])),
        ...             pa.Check(lambda s: (s == "dog").mean() > 0.5),
        ...         ]),
        ... })
        >>>
        >>> schema_withchecks.validate(t)[["probability", "category"]]
           probability category
        0         0.10      dog
        1         0.40      dog
        2         0.52      cat
        3         0.23     duck
        4         0.80      dog
        5         0.76      dog
        """

        return self.get_backend(check_obj).validate(
            check_obj=check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    @_DataFrameSchema.dtype.setter  # type: ignore[attr-defined]
    def dtype(self, value) -> None:
        """Set the dtype property."""
        self._dtype = ibis_engine.Engine.dtype(value) if value else None
