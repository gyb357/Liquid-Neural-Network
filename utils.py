from typing import Any, Dict
import yaml


@staticmethod
def ternary_operation(a: bool, b: Any, c: Any) -> Any:
    return b if a is True else c

@staticmethod
def ternary_operation_elif(a: bool, b: Any, c: bool, d: Any, e: Any) -> Any:
    return b if a is True else ternary_operation(c, d, e)

@staticmethod
def load_configs(path: str) -> Dict[str, Any]:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

