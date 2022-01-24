"""Helper functions for the FastAPI integration."""

import pandas as pd


def to_json_schema(dataframe_schema):
    """Serialize schema metadata into json-schema format.

    :param dataframe_schema: schema to write to json-schema format.

    .. note::

        This function is currently does not fully specify a pandera schema,
        and is primarily used internally to render OpenAPI docs via the
        FastAPI integration.
    """
    empty = pd.DataFrame(columns=dataframe_schema.columns.keys()).astype(
        {k: v.type for k, v in dataframe_schema.dtypes.items()}
    )
    table_schema = pd.io.json.build_table_schema(empty)

    def _field_json_schema(field):
        return {
            "type": "array",
            "items": {"type": field["type"]},
        }

    return {
        "title": dataframe_schema.name or "pandera.DataFrameSchema",
        "type": "object",
        "properties": {
            field["name"]: _field_json_schema(field)
            for field in table_schema["fields"]
        },
    }
