import yaml


# Defines config class created from yaml file
class Config:
    def __init__(self, config_file):
        config = yaml.safe_load(open(config_file, 'r'))
        self.data_path = config['data']['path']
        self.model_name = config['model']['name']
        self.load = config['model']['load']
        self.load_path = config['model']['load_path']
