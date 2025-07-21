import yaml
import os
import warnings
from typing import Any, Dict, Optional
from multiprocessing import Pool, cpu_count, get_context
import pandas as pd
import numpy as np
from functools import wraps
import dill
import time

class Config:
    def __init__(self, directory: str, filename=None):
        # If only one parameter is provided, it's the file path
        if filename is None:
            file_path = directory
        else:
        # If both parameters are provided, combine them to form the file path
            file_path = f'{directory}/{filename}'
        self.__dict__['_file_path'] = file_path  # Store the file path in the object's dict
        self.__dict__['_data'] = self._read_yaml()  # Store the config data in the object's dict

    def _read_yaml(self) -> Dict[str, Any]:
        """Reads the YAML file and returns its contents as a dictionary."""
        if not os.path.exists(self._file_path):
            warnings.warn("File path doesn't exist", UserWarning)
            return {}
        with open(self._file_path, 'r') as file:
            content = yaml.safe_load(file)

            if content is None:
                warnings.warn("The YAML file is empty or contains no data.", UserWarning)
                return {}

            return content

    def save(self) -> None:
        """Writes the current data to the YAML file."""
        with open(self._file_path, 'w') as file:
            yaml.dump(self._data, file, default_flow_style=False)

    def __getattr__(self, key: str) -> Any:
        """Gets a value using attribute-style access."""
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Sets a value using attribute-style access and saves the file."""
        self._data[key] = value
        self.save()

    def __delattr__(self, key: str) -> None:
        """Deletes a key using attribute-style access and saves the file."""
        if key in self._data:
            del self._data[key]
            self.save()

    def has_key(self, key: str) -> bool:
        """Checks if a specific key exists in the YAML file data."""
        return key in self._data

    def update(self, updates: Dict[str, Any]) -> None:
        """Updates multiple values in the YAML file data and saves the file."""
        self._data.update(updates)
        self.save()

    def clear(self) -> None:
        """Clears all data in the YAML file."""
        self._data = {}
        self.save()

    def reload(self) -> None:
        """Reloads the data from the YAML file."""
        self._data = self._read_yaml()
