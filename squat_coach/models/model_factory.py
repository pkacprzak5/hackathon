"""Model registry and factory."""
from typing import Type
from squat_coach.models.temporal_base import TemporalModelBase

_REGISTRY: dict[str, Type[TemporalModelBase]] = {}

def register_model(name: str):
    """Decorator to register a temporal model class."""
    def decorator(cls: Type[TemporalModelBase]) -> Type[TemporalModelBase]:
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> TemporalModelBase:
    """Create a model by name from the registry."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)

def available_models() -> list[str]:
    return list(_REGISTRY.keys())
