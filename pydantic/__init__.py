"""Minimal stub of the :mod:`pydantic` API for unit tests.

This lightweight implementation provides only the features used by the
AnyAI Insight Agent test-suite.  It deliberately omits validation logic and
advanced behaviours present in the real library but maintains the familiar
interface so the application code can be exercised without installing the
external dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = [
    "BaseModel",
    "Field",
    "confloat",
    "conint",
    "model_validator",
]


class _Missing:
    pass


MISSING = _Missing()


@dataclass
class _FieldInfo:
    default: Any = MISSING
    default_factory: Optional[Callable[[], Any]] = None


def Field(
    default: Any = MISSING,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    **_kwargs: Any,
) -> _FieldInfo:
    return _FieldInfo(default=default, default_factory=default_factory)


def conint(**_kwargs: Any) -> type[int]:
    return int


def confloat(**_kwargs: Any) -> type[float]:
    return float


def model_validator(*, mode: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__pydantic_validator__", mode)
        return func

    return decorator


class _BaseModelMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: Dict[str, Any]):
        annotations: Dict[str, Any] = namespace.get("__annotations__", {})
        field_info: Dict[str, _FieldInfo] = {}
        validators: Dict[str, list[Callable[[Any], Any]]] = {}

        # Collect validators defined on this class.
        for attr_name, value in list(namespace.items()):
            if callable(value) and hasattr(value, "__pydantic_validator__"):
                mode = getattr(value, "__pydantic_validator__")
                validators.setdefault(mode, []).append(value)

        for field_name in annotations:
            if field_name.startswith("__pydantic_"):
                continue
            default = namespace.get(field_name, MISSING)
            if isinstance(default, _FieldInfo):
                field_info[field_name] = default
                namespace.pop(field_name)
            else:
                field_info[field_name] = _FieldInfo(default=default)

        cls = super().__new__(mcls, name, bases, namespace)
        base_fields: Dict[str, _FieldInfo] = {}
        base_validators: Dict[str, list[Callable[[Any], Any]]] = {}
        for base in reversed(cls.__mro__[1:]):
            base_fields.update(getattr(base, "__pydantic_fields__", {}))
            for mode, funcs in getattr(base, "__pydantic_validators__", {}).items():
                base_validators.setdefault(mode, []).extend(funcs)

        base_fields.update(field_info)
        for mode, funcs in validators.items():
            base_validators.setdefault(mode, []).extend(funcs)

        cls.__pydantic_fields__ = base_fields
        cls.__pydantic_validators__ = base_validators
        return cls


class BaseModel(metaclass=_BaseModelMeta):
    __pydantic_fields__: Dict[str, _FieldInfo]
    __pydantic_validators__: Dict[str, list[Callable[[Any], Any]]]

    def __init__(self, **data: Any) -> None:
        values: Dict[str, Any] = {}
        for name, info in self.__pydantic_fields__.items():
            if name in data:
                value = data.pop(name)
            elif info.default_factory is not None:
                value = info.default_factory()
            elif info.default is not MISSING:
                value = info.default
            else:
                raise TypeError(f"Missing required field '{name}'")
            values[name] = value

        if data:
            unexpected = ", ".join(sorted(data.keys()))
            raise TypeError(f"Unexpected fields provided: {unexpected}")

        for name, value in values.items():
            setattr(self, name, value)

        for validator in self.__pydantic_validators__.get("after", []):
            result = validator(self)
            if result is not None:
                self = result  # type: ignore[assignment]

    def model_dump(
        self,
        *,
        exclude_none: bool = False,
        exclude_defaults: bool = False,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        del mode
        result: Dict[str, Any] = {}
        for name, info in self.__pydantic_fields__.items():
            value = getattr(self, name)
            if exclude_none and value is None:
                continue
            if exclude_defaults:
                default = None
                if info.default is not MISSING:
                    default = info.default_factory() if info.default_factory else info.default
                if value == default:
                    continue
            result[name] = value
        return result

    def model_dump_json(self, **json_kwargs: Any) -> str:
        data = self.model_dump(
            exclude_none=json_kwargs.pop("exclude_none", False),
            exclude_defaults=json_kwargs.pop("exclude_defaults", False),
        )
        return json.dumps(data, **json_kwargs)

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__pydantic_fields__)
        return f"{self.__class__.__name__}({fields})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseModel):
            return NotImplemented
        return self.model_dump() == other.model_dump()


