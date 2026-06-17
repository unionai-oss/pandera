---
file_format: mystnb
---

```{currentmodule} pandera
```

(supported-lib-geopandas)=

# Data Validation with GeoPandas

*new in 0.9.0*

[GeoPandas](https://geopandas.org/en/stable/docs.html) is an extension of Pandas that adds
support for geospatial data. You can use pandera to validate {py:class}`geopandas.GeoDataFrame`
and {py:class}`geopandas.GeoSeries` objects directly.

## Usage

Install `pandera` with the `geopandas` extra:

```bash
pip install 'pandera[geopandas]'
```

Import :mod:`pandera.geopandas` as a single entry point: it includes everything
from :mod:`pandera.pandas` (``Column``, ``Check``, ``Field``, ``DataFrameModel``,
dtypes, decorators, etc.) plus ``GeoDataFrameSchema`` and ``GeoDataFrameModel``.

```python
import pandera.geopandas as pg
```

For object-based validation, use {py:class}`~pandera.geopandas.GeoDataFrameSchema`
so :meth:`~pandera.api.geopandas.container.GeoDataFrameSchema.validate`
returns a {py:class}`geopandas.GeoDataFrame`. For the {ref}`class-based API <dataframe-models>`,
subclass {py:class}`~pandera.geopandas.GeoDataFrameModel` or use
{py:class}`~pandera.geopandas.DataFrameModel` when a plain :class:`pandas.DataFrame`
return type is enough.

```{code-cell} python
import geopandas as gpd
import pandera.geopandas as pg
from shapely.geometry import Polygon

geo_schema = pg.GeoDataFrameSchema({
    "geometry": pg.Column("geometry"),
    "region": pg.Column(str),
})

geo_df = gpd.GeoDataFrame({
    "geometry": [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0)))
    ],
    "region": ["NA", "SA"]
})

geo_schema.validate(geo_df)
```

You can also use the `GeometryDtype` data type in either instantiated or
un-instantiated form:

```{code-cell} python
import geopandas as gpd
import pandera.geopandas as pg

# Use ``GeometryDtype`` or ``GeometryDtype()`` interchangeably here.
geo_schema = pg.DataFrameSchema({
    "geometry": pg.Column(gpd.array.GeometryDtype()),
})
```

## `GeoDataFrameSchema`

{py:class}`~pandera.geopandas.GeoDataFrameSchema` accepts the same arguments as
{py:class}`~pandera.api.pandas.container.DataFrameSchema` but coerces the validated
result to a {py:class}`geopandas.GeoDataFrame` when it would otherwise be a plain
:class:`pandas.DataFrame`. See also :ref:`api-geopandas`.

## `GeoDataFrameModel`

Subclass {py:class}`~pandera.geopandas.GeoDataFrameModel` instead of
{py:class}`~pandera.api.pandas.model.DataFrameModel` when you want
{meth}`~pandera.api.geopandas.model.GeoDataFrameModel.validate` (and
{meth}`~pandera.api.geopandas.model.GeoDataFrameModel.example`,
{meth}`~pandera.api.geopandas.model.GeoDataFrameModel.empty`) to return a
{py:class}`geopandas.GeoDataFrame` even if the input was a plain
:class:`pandas.DataFrame`. Field definitions, checks, and `Config` behave the same
as for `DataFrameModel`; only the return type is coerced for downstream
geospatial workflows.

```{code-cell} python
import pandas as pd
import pandera.geopandas as pg
from shapely.geometry import Polygon

from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries


class GeoSchema(pg.GeoDataFrameModel):
    geometry: GeoSeries
    region: Series[str]

    class Config:
        coerce = True


gdf_in = pd.DataFrame(
    {
        "geometry": [
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
        ],
        "region": ["NA", "SA"],
    }
)
validated = GeoSchema.validate(gdf_in)
type(validated).__name__
```

## Validate on initialization

Use the {py:class}`~pandera.typing.geopandas.GeoDataFrame` generic with either
``DataFrameModel`` or ``GeoDataFrameModel``:

```{code-cell} python
import pandera.geopandas as pg
from shapely.geometry import Polygon

from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


class Schema(pg.DataFrameModel):
    geometry: GeoSeries
    region: Series[str]


df = GeoDataFrame[Schema](
    {
        'geometry': [
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0)))
        ],
        'region': ['NA','SA']
    }
)
df
```
