"""Compatibility for Python callers using the former oracle terminology."""

from functools import wraps


def legacy_keywords(**aliases):
    """Accept old keyword spellings without changing the canonical signature."""
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"use either {new!r} or legacy {old!r}, not both")
                    kwargs[new] = kwargs.pop(old)
            return function(*args, **kwargs)
        return call
    return decorate


def legacy_fields(**aliases):
    """Keep old constructor keywords and attribute access for renamed fields."""
    def decorate(cls):
        cls.__init__ = legacy_keywords(**aliases)(cls.__init__)
        for old, new in aliases.items():
            setattr(cls, old, property(
                lambda self, name=new: getattr(self, name),
                lambda self, value, name=new: setattr(self, name, value),
            ))
        return cls
    return decorate
