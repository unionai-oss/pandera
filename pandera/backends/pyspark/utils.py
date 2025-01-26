"""pyspark backend utilities."""

from pandera.config import get_config_context, get_config_global


def convert_to_list(*args):
    """Converts arguments to a list"""
    converted_list = []
    for arg in args:
        if isinstance(arg, list):
            converted_list.extend(arg)
        else:
            converted_list.append(arg)

    return converted_list


def get_full_table_validation():
    """
    Get the full table validation configuration.
    - By default, full table validation is disabled for pyspark dataframes for performance reasons.
    """
    config_global = get_config_global()
    config_ctx = get_config_context(full_table_validation_default=None)

    if config_ctx.full_table_validation is not None:
        # use context configuration if specified
        return config_ctx.full_table_validation

    if config_global.full_table_validation is not None:
        # use global configuration if specified
        return config_global.full_table_validation

    # full table validation is disabled by default for pyspark dataframes
    return False
