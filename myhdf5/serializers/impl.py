import json

import h5py
import numpy as np

from myhdf5.abc import BaseSerializer, ReIterable
from myhdf5.exceptions import FileAlreadyClosedError, SerializeError


class Hdf5Serializer(BaseSerializer):
    name = "hdf5"
    priority = 100

    @staticmethod
    def is_instance(obj):
        return isinstance(obj, (h5py.Group, h5py.Dataset))

    @staticmethod
    def serialize(grp: h5py.Group, obj):
        grp.copy(obj, "value")

    @staticmethod
    def deserialize(grp: h5py.Group):
        return grp["value"]

    @staticmethod
    def is_only_opend():
        return True


class JsonSerializer(BaseSerializer):
    name = "json"
    priority = -50

    @staticmethod
    def is_instance(obj):
        return obj is None or isinstance(
            obj, (str, int, float, bool, list, dict, set, tuple)
        )

    @staticmethod
    def serialize(grp: h5py.Group, obj):
        try:
            grp.attrs["value"] = json.dumps(obj, ensure_ascii=False, allow_nan=False)
        except json.JSONDecodeError as e:
            raise SerializeError(str(e))

    @staticmethod
    def deserialize(grp: h5py.Group):
        return json.loads(grp.attrs["value"])


class NdarraySerializer(BaseSerializer):
    name = "ndarray"
    priority = -50

    @staticmethod
    def is_instance(obj):
        return isinstance(obj, np.ndarray)

    @staticmethod
    def serialize(grp, obj: h5py.Group):
        grp.attrs["value"] = obj

    @staticmethod
    def deserialize(grp: h5py.Group):
        return grp.attrs["value"]


class DataframeSerializer(BaseSerializer):
    name = "dataframe"
    priority = -50


class ByteSerializer(BaseSerializer):
    name = "bytes"
    priority = -50

    @staticmethod
    def is_instance(obj):
        import io

        if isinstance(obj, (bytes, io.BytesIO)):
            return True

        if isinstance(obj, io.BufferedIOBase):
            return True

        return False

    @classmethod
    def serialize(cls, grp: h5py.Group, obj):
        import io

        _chunksize = 1024 * 32  # max 32k
        chunksize = int(_chunksize)
        if _chunksize != chunksize:
            raise Exception()

        def chunk_iobuffer(obj, chunksize):
            while True:
                buf = obj.read(chunksize)
                if buf:
                    yield buf
                else:
                    break

        def chunk_bytes_like(obj, chunksize):
            i = -1
            while True:
                i += 1
                from_, to = int(i * chunksize), int((i + 1) * chunksize)
                buf = obj[from_:to]
                if buf:
                    yield buf
                else:
                    break

        def chunk_memory_view(obj, chunksize):
            obj: memoryview = obj.getbuffer()
            i = -1
            while True:
                i += 1
                from_, to = int(i * chunksize), int((i + 1) * chunksize)
                buf = obj[from_:to].tobytes()
                if buf:
                    yield buf
                else:
                    break

        if isinstance(obj, bytes):
            it = chunk_bytes_like(obj, chunksize=chunksize)
        elif isinstance(obj, io.BytesIO):
            it = chunk_memory_view(obj, chunksize=chunksize)
        elif isinstance(obj, (io.BufferedIOBase)):
            it = chunk_iobuffer(obj, chunksize=chunksize)
        else:
            raise SerializeError()

        cls._create_datasets(grp, chunksize=chunksize, chunks=it)

    @staticmethod
    def _create_datasets(grp: h5py.Group, chunksize, chunks):
        grp.attrs["chunksize"] = chunksize
        MAX_ROWS = 1000000000
        FILL = len(str(MAX_ROWS))
        for i, row in enumerate(chunks):
            if not isinstance(row, bytes):
                raise TypeError()
            ds = grp.create_dataset(str(i).zfill(FILL), dtype="V1")
            ds.attrs["value"] = np.void(row)
        if i > MAX_ROWS:
            raise ValueError()

    @staticmethod
    def deserialize(grp: h5py.Group):
        func = lambda: raise_if_close(grp) and (
            x.attrs["value"].tobytes() for x in grp.values()
        )
        return ReIterable(func)

    @staticmethod
    def is_only_opend():
        return True


class WieghtsSerializer(BaseSerializer):
    name = "List[ndarray]"
    priority = -100
    track_order = True

    @staticmethod
    def is_instance(obj):
        if isinstance(obj, (str, bytes)):
            return False

        if isinstance(obj, (set, dict, tuple)):
            return False

        if hasattr(obj, "__iter__"):
            return True

        return False

    @staticmethod
    def serialize(grp: h5py.Group, obj):
        i = -1
        MAX_ROWS = 1000000000
        FILL = len(str(MAX_ROWS))
        for i, row in enumerate(obj):
            if not isinstance(row, np.ndarray):
                raise TypeError()
            grp.create_dataset(str(i).zfill(FILL), data=row)
        if i > MAX_ROWS:
            raise ValueError()

    @staticmethod
    def deserialize(grp: h5py.Group):
        func = lambda: raise_if_close(grp) and (x[::] for x in grp.values())
        return ReIterable(func)


def raise_if_close(grp: h5py.Group):
    is_closed = False
    try:
        grp.file
    except BaseException as e:
        is_closed = True

    if is_closed:
        raise FileAlreadyClosedError(f"Object is already closed")

    return True
