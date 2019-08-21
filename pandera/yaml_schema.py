"""Functions to load schemata from yaml"""

import pandera
import yaml


def df_schema_from_yaml(yaml_definition: str):
    """Create a pandera.DataFrameSchema from a yaml file"""
    schema_dict = yaml.safe_load(yaml_definition)

    # First validate if the dataframe key exists
    if 'dataframe' not in schema_dict:
        raise pandera.SchemaDefinitionError(
            "Expected key 'dataframe' missing in yaml schema definition."
        )

    # Deserialize the dataframe wide Check objects
    if schema_dict['dataframe'].get('checks'):
        df_checks = []
        for check_params in schema_dict['dataframe']['checks']:
            if not isinstance(check_params, dict):
                raise pandera.SchemaDefinitionError(
                    "Invalid definition of dataframe check: %s\n"
                    "Specify parameters for constructing a Check object." %
                    check_params
                )
            df_checks.append(pandera.Check(**check_params))
        schema_dict['dataframe']['checks'] = df_checks
    else:
        schema_dict['dataframe']['checks'] = None

    # Deserialize the Column objects
    for name, params in schema_dict['dataframe']['columns'].items():
        if 'dtype' not in params:
            raise pandera.SchemaDefinitionError(
                "Column definition '%s' lacks a data type definition. "
                "Expected key 'dtype' in the column definition." % name)
        try:
            params['dtype'] = pandera.PandasDtype[params['dtype']]
        except KeyError:
            raise pandera.SchemaDefinitionError(
                "Unknown data type '%s' in column "
                "definition '%s'" % (params['dtype'], name)
            )
        schema_dict['dataframe']['columns'][name] = pandera.Column(**params)

    return pandera.DataFrameSchema(**schema_dict['dataframe'])
