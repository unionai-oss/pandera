# Data Validation with GeoPandas

*new in 0.9.0*

[GeoPandas](https://geopandas.org/en/stable/docs.html) is an extension of Pandas that adds
support for geospatial data. You can use pandera to validate `geopandas.GeoDataFrame`
and `geopandas.GeoSeries` objects directly.

## Usage

Install `pandera` with the `geopandas` extra:

```bash
pip install 'pandera[geopandas]'
```

Import `pandera.geopandas` as a single entry point: it includes everything
from `pandera.pandas` (`Column`, `Check`, `Field`, `DataFrameModel`,
dtypes, decorators, etc.) plus `GeoDataFrameSchema` and `GeoDataFrameModel`.

```python
import pandera.geopandas as pg
```

For object-based validation, use [`GeoDataFrameSchema`](./reference/generated/pandera.geopandas.GeoDataFrameSchema.html)
so [`validate`](./reference/generated/pandera.api.geopandas.container.GeoDataFrameSchema.html)
returns a `geopandas.GeoDataFrame`. For the [class-based API](./dataframe_models.html),
subclass [`GeoDataFrameModel`](./reference/generated/pandera.geopandas.GeoDataFrameModel.html) or use
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) when a plain `pandas.DataFrame`
return type is enough.

```python
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

```python
import geopandas as gpd
import pandera.geopandas as pg

# Use ``GeometryDtype`` or ``GeometryDtype()`` interchangeably here.
geo_schema = pg.DataFrameSchema({
    "geometry": pg.Column(gpd.array.GeometryDtype()),
})
```

## `GeoDataFrameSchema`

[`GeoDataFrameSchema`](./reference/generated/pandera.geopandas.GeoDataFrameSchema.html) accepts the same arguments as
[`DataFrameSchema`](./reference/generated/pandera.api.pandas.container.DataFrameSchema.html) but coerces the validated
result to a `geopandas.GeoDataFrame` when it would otherwise be a plain
`pandas.DataFrame`. See also the [GeoPandas API reference](./reference/geopandas.html).

## `GeoDataFrameModel`

Subclass [`GeoDataFrameModel`](./reference/generated/pandera.geopandas.GeoDataFrameModel.html) instead of
[`DataFrameModel`](./reference/generated/pandera.api.pandas.model.DataFrameModel.html) when you want
[`validate`](./reference/generated/pandera.geopandas.GeoDataFrameModel.html) (and
[`example`](./reference/generated/pandera.geopandas.GeoDataFrameModel.html),
[`empty`](./reference/generated/pandera.geopandas.GeoDataFrameModel.html)) to return a
`geopandas.GeoDataFrame` even if the input was a plain
`pandas.DataFrame`. Field definitions, checks, and `Config` behave the same
as for `DataFrameModel`; only the return type is coerced for downstream
geospatial workflows.

```python
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

Use the [`GeoDataFrame`](./reference/generated/pandera.typing.geopandas.GeoDataFrame.html) generic with either
`DataFrameModel` or `GeoDataFrameModel`:

```python
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
