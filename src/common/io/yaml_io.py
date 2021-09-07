import yaml


def yaml_reader(yaml_path: str):
    """
    Load the yaml file from disk.
    :param yaml_path: Path to configuration file.
    :return:
    configuration dictionary.
    """
    with open(yaml_path, 'r') as config_file:
        config = yaml.load(config_file, yaml.UnsafeLoader)
    return config
