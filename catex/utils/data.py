import functools
import inspect
from itertools import chain
from contextlib import contextmanager
from typing import Generator, Any, Callable, Type, Set
from torch.utils.data import DataLoader

#  These functions come from pytorch_lighting/utilities/data.py

# https://stackoverflow.com/a/63851681/9201239
def _get_all_subclasses(cls: Type[Any]) -> Set[Type[Any]]:
    """Returns a list of all classes that inherit directly or indirectly from the given class."""
    subclasses = set()

    def recurse(cl: Type[Any]) -> None:
        for subclass in cl.__subclasses__():
            subclasses.add(subclass)
            recurse(subclass)

    recurse(cls)
    return subclasses


@contextmanager
def _replace_dataloader_init_method() -> Generator[None, None, None]:
    """This context manager is used to add support for re-instantiation of custom (subclasses) of
    :class:`~torch.utils.data.DataLoader`. It patches the ``__init__`` method."""
    subclasses = _get_all_subclasses(DataLoader)
    for subclass in subclasses:
        subclass._old_init = subclass.__init__
        subclass.__init__ = _wrap_init(subclass.__init__)
    yield
    for subclass in subclasses:
        subclass.__init__ = subclass._old_init
        del subclass._old_init

def _wrap_init(init: Callable) -> Callable:
    """Wraps the ``__init__`` method of the dataloader in order to enable re-instantiation of custom subclasses of
    :class:`~torch.utils.data.DataLoader`."""

    @functools.wraps(init)
    def wrapper(obj: DataLoader, *args: Any, **kwargs: Any) -> None:
        params = dict(inspect.signature(obj.__init__).parameters)
        params.pop("args", None)
        params.pop("kwargs", None)
        cls = type(obj)
        for arg_name, arg_value in chain(zip(params, args), kwargs.items()):
            if hasattr(cls, arg_name) and getattr(cls, arg_name).fset is None:
                # the class defines a read-only (no setter) property of this name. it's likely that the implementation
                # will set `self._arg_name = arg_value` in `__init__` which is the attribute returned by the `arg_name`
                # property so we are fine skipping in that case
                continue
            setattr(obj, arg_name, arg_value)
        init(obj, *args, **kwargs)

    return wrapper