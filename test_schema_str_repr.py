import pandera as pa

multiindex = pa.MultiIndex(
    indexes=[
        pa.Index(int, name=0),
        pa.Index(float, name=1),
        pa.Index(str, name=2),
    ]
)

schema = pa.DataFrameSchema(
    columns={
        "col1": pa.Column(int),
        "col2": pa.Column(str),
        "col3": pa.Column(float),
    },
    checks=[pa.Check.gt(0), pa.Check.lt(10)],
    index=multiindex,
    coerce=True,
    strict=True,
)

print(schema.__repr__())
print(schema)

series_schema = pa.SeriesSchema(int)
# print(series_schema)
