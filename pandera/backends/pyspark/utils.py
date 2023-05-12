"""pyspark backend utilities."""
import yaml
import os
import pandera


def convert_to_list(*args):
    converted_list = []
    for arg in args:
        if isinstance(arg, list):
            converted_list.extend(arg)
        else:
            converted_list.append(arg)

    return converted_list


class ConfigParams(dict):
    def __init__(self, module_name, config_name):
        self.module_name = module_name
        self.config_name = config_name
        self.config = self.fetch_yaml(self.module_name, self.config_name)
        self.validate_params(self.config)
        super().__init__(self.config)

    @staticmethod
    def fetch_yaml(module_name, config_file):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(pandera.__file__), ".."))
        path = os.path.join(root_dir, 'conf', module_name, config_file)
        with open(path) as file:
            return yaml.safe_load(file)

    @staticmethod
    def validate_params(config):
        if not config.get('VALIDATION'):
            raise ValueError('Parameter "VALIDATION" not found in config, ensure the parameter value is in upper case')
        else:
            if config.get('VALIDATION') not in ['ENABLE', 'DISABLE']:
                raise ValueError("Parameter 'VALIDATION' only supports 'ON' or 'OFF' as valid values."
                                 "Ensure the value is in upper case only")
        if not config.get('DEPTH'):
            raise ValueError('Parameter "DEPTH" not found in config, ensure the parameter value is in upper case')
        else:
            if config.get('DEPTH') not in ['SCHEMA_ONLY', 'DATA_ONLY', 'SCHEMA_AND_DATA']:
                raise ValueError("Parameter 'VALIDATION' only supports 'ON' or 'OFF' as valid values."
                                 "Ensure the value is in upper case only")

