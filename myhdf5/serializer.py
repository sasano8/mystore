import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Set, Type

import h5py
from genericpath import isdir

from .abc import BaseSerializer, extensionmethod
from .appname import APPNAME
from .directory import InfinityTempNames, RealDir
from .serializers import (
    ByteSerializer,
    Hdf5Serializer,
    JsonSerializer,
    NdarraySerializer,
    WieghtsSerializer,
)


class Serializer:
    def __init__(self, *serializers: Type[BaseSerializer]):
        tmp_dic = {}
        [self._add_serializer(tmp_dic, x) for x in serializers]
        self._serializers = self._sort_by_priority(tmp_dic)

    @staticmethod
    def _add_serializer(dic: dict, serializer):
        if not issubclass(serializer, BaseSerializer):
            raise TypeError()

        if not (
            hasattr(serializer, "name")
            and hasattr(serializer, "priority")
            and hasattr(serializer, "track_order")
        ):
            raise TypeError()

        dic.setdefault(serializer.priority, []).append(serializer)

    @staticmethod
    def _sort_by_priority(dic: dict):
        items = sorted(dic.items(), key=lambda x: x[0], reverse=True)
        return dict(items)

    def get_serializer_by_value(self, value) -> "BaseSerializer | None":
        for priority, serializers in self._serializers.items():
            for cls in serializers:
                if cls.is_instance(value):
                    return cls

        return None

    def get_serializer_by_src(self, src: h5py.Group) -> "BaseSerializer | None":
        for priority, serializers in self._serializers.items():
            for cls in serializers:
                if cls.name == src.attrs["name"]:
                    return cls

        return None

    @staticmethod
    def _save(dest, obj, serializer, meta=None):
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise TypeError("meta must be dict.")
        dest.attrs["appname"] = APPNAME
        dest.attrs["name"] = serializer.name
        updated_at = utc_now_isoformat()
        if "created_at" not in dest.attrs:
            dest.attrs["created_at"] = updated_at
        dest.attrs["updated_at"] = updated_at
        dest.attrs["meta"] = json.dumps(meta)
        serializer.serialize(dest, obj)

    def save(self, dest, obj, *, meta=None, serializer=None, overwrite: bool = False):
        if serializer is None:
            serializer = self.get_serializer_by_value(obj)
        if serializer is None:
            raise Exception()

        mode = "w" if overwrite else "w-"

        try:
            from_path, _dest = self._is_valid_dest(
                dest, mode, track_order=serializer.track_order
            )
        except FileExistsError as e:
            raise FileExistsError(
                "File already exists. If you want to overwrite, set `overwrite=True`"
            )

        if not is_empty_group(_dest):
            raise Exception()

        if from_path:
            with _dest as _dest:
                self._save(_dest, obj, serializer, meta=meta)
        else:
            self._save(_dest, obj, serializer, meta=meta)

        return dest

    def save_file(
        self, dest, file_path, mode="rb", *, meta=None, overwrite: bool = False
    ):
        if not isinstance(file_path, (str, Path)):
            raise Exception()

        with open(file_path, mode) as f:
            return self.save(dest, f, meta=meta, overwrite=overwrite)

    def save_weights(self, dest, obj, *, meta=None, overwrite: bool = False):
        return self.save(
            dest, obj, serializer=WieghtsSerializer, meta=meta, overwrite=overwrite
        )

    def load(self, src, map=None):
        from_path, _src = self._is_valid_dest(src, "r")
        serializer = self.get_serializer_by_src(_src)
        if serializer is None:
            raise Exception()

        if serializer.is_only_opend() and map is None:
            if from_path:
                raise Exception("srcはopen状態のhdf5である必要があります")

        map = map or (lambda x: x)

        if from_path:
            with _src as _src:
                result = serializer.deserialize(_src)
                return map(result)
        else:
            result = serializer.deserialize(_src)
            return map(result)

    @contextmanager
    def load_with(self, src):
        if not isinstance(src, (str, Path)):
            raise Exception()

        with h5py.File(src, "r") as f:
            yield self.load(f)

    def load_meta(self, src):
        dic = self.load_info(src, attrs=["meta"])
        return dic["meta"]

    def load_info(
        self, src, attrs=["appname", "name", "created_at", "updated_at", "meta"]
    ):
        from_path, _src = self._is_valid_dest(src, "r")
        dic = {k: _src.attrs[k] for k in attrs}
        if "meta" in dic:
            dic["meta"] = json.loads(dic["meta"])
        return dic

    def _is_valid_dest(
        self, dest, mode, track_order: bool = False
    ) -> "str | h5py.File | h5py.Group":
        from_path = False
        if isinstance(dest, (str, Path)):
            from_path = True
            dest = h5py.File(dest, mode, track_order=track_order)

        if not isinstance(dest, (h5py.Group, h5py.File)):
            raise Exception()

        return from_path, dest


def is_empty_group(grp: h5py.Group):
    return is_empty_info(grp) and is_empty_datasets(grp)


def is_empty_info(grp: h5py.Group):
    return len(grp.attrs) == 0


def is_empty_datasets(grp: h5py.Group):
    return len(grp) == 0


def utc_now_isoformat():
    return datetime.now(timezone.utc).isoformat()
