"""Schema components for xarray (`Coordinate`, `DataVar`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pandera.api.base.types import CheckList, ParserList, StrictType
from pandera.api.checks import Check
from pandera.api.hypotheses import Hypothesis
from pandera.api.parsers import Parser
from pandera.errors import SchemaDefinitionError

if TYPE_CHECKING:
    from pandera.api.xarray.container import DataArraySchema


class Coordinate:
    """Specification for one entry in ``DataArraySchema(coords=...)``.

    Coordinates are validated like small
    :class:`~pandera.api.xarray.container.DataArraySchema` instances (dtype,
    nullability, checks, parsers).

    Parameters
    ----------
    dtype
        Expected value dtype (via the xarray dtype engine).
    dims
        Expected dimension names of the coordinate variable.
    dimension
        If True, the coordinate is a dimension coordinate (indexes its dim).
    required
        If True (default), the coordinate must exist. If False, the
        coordinate is optional; when present all other constraints apply.
    indexed
        If True, coordinate values must match the dimension index (see backend).
    checks, parsers
        Applied to the coordinate :class:`~xarray.DataArray`.
    nullable, coerce
        Same meaning as on :class:`DataArraySchema`.
    strict_coords, strict_attrs
        Passed through when materializing a
        :class:`~pandera.api.xarray.container.DataArraySchema`.
    name, title, description, metadata
        Documentation / identification metadata.
    """

    def __init__(
        self,
        dtype: Any | None = None,
        dims: tuple[str, ...] | None = None,
        dimension: bool | None = None,
        required: bool = True,
        checks: CheckList | None = None,
        parsers: ParserList | None = None,
        nullable: bool = False,
        coerce: bool = False,
        indexed: bool | None = None,
        name: str | None = None,
        strict_coords: StrictType = False,
        strict_attrs: StrictType = False,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]
        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]
        self.dtype = dtype
        self.dims = dims
        self.dimension = dimension
        self.required = required
        self.checks = checks
        self.parsers = parsers
        self.nullable = nullable
        self.coerce = coerce
        self.indexed = indexed
        self.name = name
        self.strict_coords = strict_coords
        self.strict_attrs = strict_attrs
        self.title = title
        self.description = description
        self.metadata = metadata

    def to_data_array_schema(self, coord_name: str) -> DataArraySchema:
        """Materialize as a :class:`DataArraySchema` for the coordinate."""
        from pandera.api.xarray.container import DataArraySchema

        return DataArraySchema(
            dtype=self.dtype,
            dims=self.dims,
            coords=None,
            attrs=None,
            name=self.name or coord_name,
            checks=self.checks,
            parsers=self.parsers,
            coerce=self.coerce,
            nullable=self.nullable,
            chunked=None,
            array_type=None,
            strict_coords=self.strict_coords,
            strict_attrs=self.strict_attrs,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )


class DataVar:
    """Per-variable schema inside :class:`~pandera.api.xarray.container.DatasetSchema`.

    ``aligned_with`` / ``broadcastable_with`` express grid relationships to
    other data variables (same shape vs broadcastable). Other fields mirror
    :class:`DataArraySchema` (dims, dtype, checks, etc.).

    Parameters
    ----------
    required
        If False, the variable may be absent; ``default`` can fill it.
    alias
        Actual name in the :class:`~xarray.Dataset` if different from the key.
    default
        Filled when ``required=False`` and the variable is missing.
    aligned_with
        Names of variables that must share dims and shape with this one.
    broadcastable_with
        Names of variables this one must be broadcastable against (see backend).
    dtype, dims, sizes, shape, coords, attrs
        Structural constraints (``sizes`` and ``shape`` are mutually exclusive).
    checks, parsers, coerce, nullable, chunked, array_type
        Same roles as on :class:`DataArraySchema`.
    strict_coords, strict_attrs
        Coordinate / attribute strictness for the variable array.
    name, title, description, metadata
        Documentation / identification metadata.
    """

    def __init__(
        self,
        *,
        required: bool = True,
        alias: str | None = None,
        regex: bool = False,
        default: Any | None = None,
        aligned_with: tuple[str, ...] | None = None,
        broadcastable_with: tuple[str, ...] | None = None,
        dtype: Any | None = None,
        dims: tuple[str | None, ...] | list[str | None] | None = None,
        ordered_dims: bool = True,
        sizes: dict[str, int | None] | None = None,
        shape: tuple[int | None, ...] | None = None,
        coords: dict[str, Any] | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        name: str | None = None,
        checks: CheckList | None = None,
        parsers: ParserList | None = None,
        coerce: bool = False,
        nullable: bool = False,
        chunked: bool | None = None,
        array_type: Any | None = None,
        strict_coords: StrictType = False,
        strict_attrs: StrictType = False,
        title: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ):
        if regex:
            raise SchemaDefinitionError(
                "DataVar(regex=True) is reserved for Phase 2 "
                "(pattern keys in DatasetSchema)."
            )
        if aligned_with and broadcastable_with:
            overlap = set(aligned_with) & set(broadcastable_with)
            if overlap:
                raise SchemaDefinitionError(
                    "aligned_with and broadcastable_with must not name "
                    f"the same variables: {sorted(overlap)}"
                )
        if checks is None:
            checks = []
        if isinstance(checks, (Check, Hypothesis)):
            checks = [checks]
        if parsers is None:
            parsers = []
        if isinstance(parsers, Parser):
            parsers = [parsers]
        if sizes is not None and shape is not None:
            raise SchemaDefinitionError(
                "Pass only one of `sizes` and `shape` on DataVar."
            )

        self.required = required
        self.alias = alias
        self.regex = regex
        self.default = default
        self.aligned_with = aligned_with
        self.broadcastable_with = broadcastable_with
        self.dtype = dtype
        self.dims = tuple(dims) if dims is not None else None
        self.ordered_dims = ordered_dims
        self.sizes = sizes
        self.shape = shape
        self.coords = coords
        self.attrs = attrs
        self.name = name
        self.checks = checks
        self.parsers = parsers
        self.coerce = coerce
        self.nullable = nullable
        self.chunked = chunked
        self.array_type = array_type
        self.strict_coords = strict_coords
        self.strict_attrs = strict_attrs
        self.title = title
        self.description = description
        self.metadata = metadata

    def to_data_array_schema(self, data_var_key: str) -> DataArraySchema:
        """Array-level validation spec (no dataset-only fields)."""
        from pandera.api.xarray.container import DataArraySchema

        return DataArraySchema(
            dtype=self.dtype,
            dims=self.dims,
            ordered_dims=self.ordered_dims,
            sizes=self.sizes,
            shape=self.shape,
            coords=self.coords,
            attrs=self.attrs,
            name=self.name or data_var_key,
            checks=self.checks,
            parsers=self.parsers,
            coerce=self.coerce,
            nullable=self.nullable,
            chunked=self.chunked,
            array_type=self.array_type,
            strict_coords=self.strict_coords,
            strict_attrs=self.strict_attrs,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )
