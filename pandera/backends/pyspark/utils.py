"""pyspark backend utilities."""


def convert_to_list(*args):
    """Converts arguments to a list"""
    converted_list = []
    for arg in args:
        if isinstance(arg, list):
            converted_list.extend(arg)
        else:
            converted_list.append(arg)

    return converted_list
