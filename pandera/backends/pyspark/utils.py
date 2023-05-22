"""pyspark backend utilities."""
import os
import warnings

DEFAULT_CONFIG = {"VALIDATION": "ENABLE", "DEPTH": "SCHEMA_AND_DATA"}


def convert_to_list(*args):
    """Converts arguments to a list"""
    converted_list = []
    for arg in args:
        if isinstance(arg, list):
            converted_list.extend(arg)
        else:
            converted_list.append(arg)

    return converted_list


class ConfigParams(dict):
    """This class inherits froma  dictionary object and holds parameters for config variable"""

    def __init__(self):
        # Default config values will run everything
        self.config = DEFAULT_CONFIG
        self.set_config()
        super().__init__(self.config)

    def set_config(self):
        """This function sets the config for the instance of config param"""
        if os.environ.get("VALIDATION"):
            self.config["VALIDATION"] = os.environ.get("VALIDATION")
            warnings.warn(
                "Setting the VALIDATION config from environment variables",
                RuntimeWarning,
                stacklevel=2,
            )

        if os.environ.get("DEPTH"):
            self.config["DEPTH"] = os.environ.get("DEPTH")
            warnings.warn(
                "Setting the DEPTH config from environment variables",
                RuntimeWarning,
                stacklevel=2,
            )

        if (not os.environ.get("VALIDATION")) and (not os.environ.get("DEPTH")):
            warnings.warn(
                "Setting the VALIDATION and DEPTH config from default values"
                " since no environment variable found to overload",
                RuntimeWarning,
            )
        self.validate_params(self.config)

    @staticmethod
    def validate_params(config):
        """This function validates the input of the config"""
        if not config.get("VALIDATION"):
            raise ValueError(
                'Parameter "VALIDATION" not found in config, ensure the parameter value is in upper case'
            )
        if config.get("VALIDATION") not in ["ENABLE", "DISABLE"]:
            raise ValueError(
                "Parameter 'VALIDATION' only supports 'ENABLE' or 'DISABLE' as valid values."
                "Ensure the value is in upper case only"
            )
        if not config.get("DEPTH"):
            raise ValueError(
                'Parameter "DEPTH" not found in config, ensure the parameter value is in upper case'
            )
        if config.get("DEPTH") not in [
            "SCHEMA_ONLY",
            "DATA_ONLY",
            "SCHEMA_AND_DATA",
        ]:
            raise ValueError(
                "Parameter 'DEPTH' only supports 'SCHEMA_AND_DATA', 'SCHEMA_ONLY' or 'DATA_ONLY' "
                "as valid values. Ensure the value is in upper case only"
            )


PANDERA_CONFIG = ConfigParams()
