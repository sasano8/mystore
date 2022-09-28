import os
from typing import TYPE_CHECKING, Set

from .abc import BaseSerializer, extensionmethod
from .directory import InfinityTempNames, RealDir
from .serializer import Serializer
from .serializers import (
    ByteSerializer,
    Hdf5Serializer,
    JsonSerializer,
    NdarraySerializer,
    WieghtsSerializer,
)

default_serializers: Set[BaseSerializer] = {
    WieghtsSerializer,
    Hdf5Serializer,
    ByteSerializer,
    JsonSerializer,
    NdarraySerializer,
}


app = Serializer(*default_serializers)


class ModelFile:
    def __init__(self, file_path):
        self.__root__ = file_path

    @extensionmethod
    def save(self: str, obj, *, meta=None, serializer=None, overwrite: bool = False):
        return app.save(
            self, obj, meta=meta, serializer=serializer, overwrite=overwrite
        )

    @extensionmethod
    def save_file(
        self: str, input_path, mode="rb", *, meta=None, overwrite: bool = False
    ):
        return app.save_file(
            self, input_path, mode=mode, meta=meta, overwrite=overwrite
        )

    @extensionmethod
    def save_weights(self: str, obj, *, meta=None, overwrite: bool = False):
        return app.save_weights(self, obj, meta=meta, overwrite=overwrite)

    @extensionmethod
    def load(self: str, map=None):
        return app.load(self, map=map)

    @extensionmethod
    def load_with(self: str):
        return app.load_with(self)

    @extensionmethod
    def load_meta(self: str):
        return app.load_meta(self)

    @extensionmethod
    def load_info(
        self: str, attrs=["appname", "name", "created_at", "updated_at", "meta"]
    ):
        return app.load_info(self, attrs=attrs)


class TempModelFile(ModelFile):
    def __init__(self):
        ...

    def __enter__(self):
        self._tmpdir = InfinityTempNames()
        self.__root__ = self._tmpdir.__enter__().next()
        return self

    def __exit__(self, *args, **kwargs):
        self._tmpdir.__exit__(*args, **kwargs)
        del self._tmpdir
        del self.__root__


class ModelStore:
    def __init__(self, path, ignore_exists: bool = False):
        self._path = RealDir(path, ignore_exists=ignore_exists)
        self._ignore_exists = ignore_exists

    def __enter__(self):
        self._tmpdir = RealDir(self._path, ignore_exists=self._ignore_exists)
        self.__root__ = self._tmpdir.__enter__().dirname
        return self

    def __exit__(self, *args, **kwargs):
        self._tmpdir.__exit__(*args, **kwargs)
        del self._tmpdir
        del self.__root__

    @staticmethod
    def _join_path(self, file):
        if not file:
            raise ValueError()

        dir = self.__root__
        joined = os.path.join(dir, file)
        if not (dir == joined[: len(dir)]):
            raise ValueError(f"{file}: {dir}")

        return os.path.abspath(joined)

    def save(self, dest, obj, *, meta=None, serializer=None, overwrite: bool = False):
        _dest = self._join_path(self, dest)
        app.save(_dest, obj, meta=meta, serializer=serializer, overwrite=overwrite)
        return _dest

    def save_file(
        self, dest, input_path, mode="rb", *, meta=None, overwrite: bool = False
    ):
        _dest = self._join_path(self, dest)
        app.save_file(_dest, input_path, mode=mode, meta=meta, overwrite=overwrite)
        return _dest

    def save_weights(self, dest, obj, *, meta=None, overwrite: bool = False):
        _dest = self._join_path(self, dest)
        app.save_weights(_dest, obj, meta=meta, overwrite=overwrite)
        return _dest

    def load(self, dest, map=None):
        _dest = self._join_path(self, dest)
        return app.load(_dest, map=map)

    def load_with(self, dest):
        _dest = self._join_path(self, dest)
        return app.load_with(self)

    def load_meta(self, dest):
        _dest = self._join_path(self, dest)
        return app.load_meta(self)

    def load_info(
        self, dest, attrs=["appname", "name", "created_at", "updated_at", "meta"]
    ):
        _dest = self._join_path(self, dest)
        return app.load_info(self, attrs=attrs)

    def list_by_updated_at(self, desc: bool = False):
        dir = os.path.abspath(self.__root__)
        files = (os.path.join(dir, f) for f in os.listdir(dir))
        files = filter(os.path.isfile, files)
        return list(sorted(files, key=lambda x: os.path.getmtime(x), reverse=desc))


class TempModelStore(ModelStore):
    def __init__(self):
        ...

    def __enter__(self):
        self._tmpdir = InfinityTempNames()
        self.__root__ = self._tmpdir.__enter__().dirname
        return self

    def __exit__(self, *args, **kwargs):
        self._tmpdir.__exit__(*args, **kwargs)
        del self._tmpdir
        del self.__root__

    def file(self, name=None, suffix=".hdf5") -> ModelFile:
        if name is None:
            return ModelFile(self._tmpdir.next(suffix=suffix))
        else:
            file = name + suffix
            return ModelFile(os.path.join(self._tmpdir.dirname, file))


if not TYPE_CHECKING:

    class ModelFile(ModelFile, str):
        ...


# class ModelExtension:
#     def __init__(self, model):
#         self.__root__ = model

#     @extensionmethod
#     def get_weights(model: Any) -> Any:
#         import torch
#         from keras.engine.functional import Functional

#         if isinstance(model, torch.nn.Module):
#             ...

#         if isinstance(model, Functional):
#             return model.get_weights()

#     @extensionmethod
#     def save_model(model: Any) -> Any:
#         import torch
#         from keras.engine.functional import Functional

#         if isinstance(model, Functional):
#             return model.save()
