"""pyspark backend utilities."""
import os
import warnings

DEFAULT_CONFIG = {"PANDERA_VALIDATION": "ENABLE", "PANDERA_DEPTH": "SCHEMA_AND_DATA"}


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
    """This class inherits from a dictionary object and holds parameters for config variable"""

    def __init__(self):
        # Default config values will run everything
        self.config = DEFAULT_CONFIG
        self.set_config()
        super().__init__(self.config)

    def set_config(self):
        """This function sets the config for the instance of config param"""
        if os.environ.get("PANDERA_VALIDATION"):
            self.config["PANDERA_VALIDATION"] = os.environ.get("PANDERA_VALIDATION")
            warnings.warn(
                "Setting the PANDERA_VALIDATION config from environment variables",
                RuntimeWarning,
                stacklevel=2,
            )

        if os.environ.get("PANDERA_DEPTH"):
            self.config["PANDERA_DEPTH"] = os.environ.get("PANDERA_DEPTH")
            warnings.warn(
                "Setting the PANDERA_DEPTH config from environment variables",
                RuntimeWarning,
                stacklevel=2,
            )

        if (not os.environ.get("PANDERA_VALIDATION")) and (
            not os.environ.get("PANDERA_DEPTH")
        ):
            warnings.warn(
                "Setting the PANDERA_VALIDATION and PANDERA_DEPTH config from default values"
                " since no environment variable found to overload",
                RuntimeWarning,
            )
        self.validate_params(self.config)

    @staticmethod
    def validate_params(config):
        """This function validates the input of the config"""
        if not config.get("PANDERA_VALIDATION"):
            raise ValueError(
                'Parameter "PANDERA_VALIDATION" not found in config, ensure the parameter value is in upper case'
            )
        if config.get("PANDERA_VALIDATION") not in ["ENABLE", "DISABLE"]:
            raise ValueError(
                "Parameter 'PANDERA_VALIDATION' only supports 'ENABLE' or 'DISABLE' as valid values."
                "Ensure the value is in upper case only"
            )
        if not config.get("PANDERA_DEPTH"):
            raise ValueError(
                'Parameter "PANDERA_DEPTH" not found in config, ensure the parameter value is in upper case'
            )
        if config.get("PANDERA_DEPTH") not in [
            "SCHEMA_ONLY",
            "DATA_ONLY",
            "SCHEMA_AND_DATA",
        ]:
            raise ValueError(
                "Parameter 'PANDERA_DEPTH' only supports 'SCHEMA_AND_DATA', 'SCHEMA_ONLY' or 'DATA_ONLY' "
                "as valid values. Ensure the value is in upper case only"
            )


PANDERA_CONFIG = ConfigParams()
