import copy
import os
import yaml
from typing import Any, IO


class Config(dict):
    """ Load YAML config file.

    All attributes are direcly accessible.

    For example, if `config.yaml` has the following section:

    ```
    dataset:
      name: Multi30k
    ```

    Then, the dataset name can be access via `config.datasest.name`.
    """
    def __init__(self, d: dict) -> None:
        super().__init__(d)
        self.d = copy.deepcopy(d) # keep the original dictionary
        for key, val in d.items():
            if isinstance(val, dict):
                val = Config(val)
            elif isinstance(val, list):
                val = [Config(x) if isinstance(x, dict) else x for x in val]
            setattr(self, key, val)
            self[key] = val

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.d, f, sort_keys=False)


def load_config(path: str) -> Config:
    yaml_str = load_yaml(path)
    return Config(yaml.safe_load(yaml_str))


def load_yaml(path: str, indent: int=0) -> Config:
    root_dir = os.path.dirname(path)
    with open(path, 'r') as f:
        s = ''
        for line in f.readlines():
            if '!include' in line:
                include_pos = line.index('!include')
                pos = include_pos + 9
                path = os.path.join(root_dir, line[pos:].strip())
                s += '\n' + load_yaml(path, indent + include_pos)
            else:
                s += ' ' * indent + line
    return s