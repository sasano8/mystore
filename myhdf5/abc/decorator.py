from functools import wraps
from types import MethodType
from typing import TypeVar

F = TypeVar("F")


class ExtensionMethod:
    def __init__(self, func):
        self.__func__ = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.__func__  # インスタンスなし時はそのまま関数を返す
        func = self.__func__

        @wraps(func)  # 関数の引数情報などを引き継ぎ
        def wrapper(self, *args, **kwargs):
            return func(
                getattr(self, "__root__"), *args, **kwargs
            )  # selfをself.__root__に置換

        return MethodType(wrapper, obj)  # インスタンスと関数をバインドしたメソッドを返す


def extensionmethod(func: F) -> F:
    return ExtensionMethod(func)  # type: ignore
