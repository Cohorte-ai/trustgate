"""Plugin interface for user-defined canonicalization."""

from __future__ import annotations

import importlib

from theaios.trustgate.canonicalize import Canonicalizer


def load_custom_canonicalizer(dotted_path: str) -> Canonicalizer:
    """Dynamically import and instantiate a user-defined canonicalizer.

    Args:
        dotted_path: Fully qualified path, e.g.
            ``"my_package.my_module.MyCanonicalizer"``

    Returns:
        An instance of the user's canonicalizer class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class doesn't exist in the module.
        TypeError: If the class is not a Canonicalizer subclass.
    """
    if "." not in dotted_path:
        raise ImportError(
            f"Invalid dotted path '{dotted_path}': expected 'module.ClassName'"
        )

    module_path, class_name = dotted_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type) or not issubclass(cls, Canonicalizer):
        raise TypeError(
            f"'{dotted_path}' is not a Canonicalizer subclass "
            f"(got {type(cls).__name__}: {cls})"
        )

    return cls()
