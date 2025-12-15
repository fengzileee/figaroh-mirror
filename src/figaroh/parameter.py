from typing import Dict, Sequence, List

import numpy as np


class Parameter:
    def __init__(self, names: Sequence[str], values: Sequence[float]):
        self._dict = dict(zip(names, values))
        self._keys = list(names)

    def __getitem__(self, name: str) -> float:
        if name in self._dict:
            return self._dict[name]
        raise KeyError(f"Parameter '{name}' not found.")

    @property
    def names(self) -> List[str]:
        return list(self._keys)

    def as_dict(self) -> Dict[str, float]:
        return self._dict.copy()

    def get_values(self) -> np.ndarray:
        self._values = tuple(self._dict[key] for key in self._keys)
        return np.array(self._values)

    def get_index(self, name) -> int:
        return self._keys.index(name)

    def get_values_by_names(self, names: list) -> np.ndarray:
        values = [self._dict[name] for name in names]
        return np.array(values)

    def update_values(self, new_values: np.ndarray):
        if len(new_values) != len(self._keys):
            raise ValueError("Length of new_values must match number of parameters.")
        for key, value in zip(self._keys, new_values):
            self._dict[key] = value

    def update_values_by_dict(self, new_values_dict: Dict[str, float]):
        for key, value in new_values_dict.items():
            if key in self._dict:
                self._dict[key] = value
            else:
                raise KeyError(f"Parameter '{key}' not found.")

    def copy(self):
        return Parameter(self._keys, self.get_values())
