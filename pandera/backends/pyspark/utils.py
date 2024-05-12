"""pyspark backend utilities."""

from pandera.config import get_config_context


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
    config = get_config_context()
    if config.full_table_validation is not None:
        return config.full_table_validation
    return False
