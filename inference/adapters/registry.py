from __future__ import annotations

from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class AdapterRegistry(Generic[T]):
    """Simple name-to-class registry for adapters."""
    def __init__(self, kind: str):
        """Create a registry for a specific adapter kind."""
        self._kind = kind
        self._registry: Dict[str, type[T]] = {}

    def register(self, name: str, adapter_cls: type[T]) -> None:
        """Register an adapter class under a name."""
        key = name.lower()
        if key in self._registry:
            raise ValueError(f"{self._kind} adapter '{name}' is already registered.")
        self._registry[key] = adapter_cls

    def get(self, name: str) -> type[T]:
        """Resolve an adapter class by name."""
        key = name.lower()
        if key not in self._registry:
            known = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Unknown {self._kind} adapter '{name}'. Known: {known}")
        return self._registry[key]

    def create(self, name: str, **kwargs) -> T:
        """Instantiate an adapter by name."""
        adapter_cls = self.get(name)
        return adapter_cls(**kwargs)


task_adapters: AdapterRegistry = AdapterRegistry("task")
metric_adapters: AdapterRegistry = AdapterRegistry("metric")
