---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: pandera
```

(supported-lib-geopandas)=

# Data Validation with GeoPandas

*new in 0.9.0*

[GeoPandas](https://geopandas.org/en/stable/docs.html) is an extension of Pandas that adds
support for geospatial data. You can use pandera to validate {py:func}`~geopandas.GeoDataFrame`
and {py:func}`~geopandas.GeoSeries` objects directly. First, install
`pandera` with the `geopandas` extra:

```bash
pip install 'pandera[geopandas]'
```

Then you can use pandera schemas to validate geodataframes. In the example
below we'll use the {ref}`class-based API <dataframe-models>` to define a
{py:class}`~pandera.api.pandas.model.DataFrameModel` for validation.

```{code-cell} python
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from shapely.geometry import Polygon

geo_schema = pa.DataFrameSchema({
    "geometry": pa.Column("geometry"),
    "region": pa.Column(str),
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
geo_schema = pa.DataFrameSchema({
    "geometry": pa.Column(gpd.array.GeometryDtype),
    # or
    "geometry": pa.Column(gpd.array.GeometryDtype()),
})
```

If you want to validate-on-instantiation, you can use the
{py:class}`~pandera.typing.geopangas.GeoDataFrame` generic type with the
dataframe model defined above:

```{code-cell} python
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


class Schema(pa.DataFrameModel):
    geometry: GeoSeries
    region: Series[str]


# create a geodataframe that's validated on object initialization
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
