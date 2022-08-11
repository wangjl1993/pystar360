import os
from importlib import import_module
from typing import List, Union

from omegaconf import DictConfig, ListConfig
from torch import load

from pystar360.ano.lib1.models.components import AnomalyModule

def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
   
    torch_model_list: List[str] = ["pat"]
    model: AnomalyModule

    if config.model.name in torch_model_list:
        module = import_module(f"pystar360.ano.lib1.models.{config.model.name}")
        model = getattr(module, f"{config.model.name.capitalize()}Lightning")
    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    model = model(config)

    return model
