class ConfigHolder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, config_params=None):
        self.config = config_params or {}

    def get_parameter(self, key):
        return self.config.get(key)

    def set_parameter(self, key, value):
        self.config[key] = value

    def set_config(self, config):
        self.config = config


config_holder = ConfigHolder()
