import logging
import os

from box import ConfigBox
from ruamel.yaml import YAML


class ParamsProvider:
    def __init__(self, params_path = "params.yaml"):
        yaml = YAML(typ='safe')
        self.params = ConfigBox(yaml.load(open(params_path, encoding='utf-8')))

    def get_params(self):
        return self.params