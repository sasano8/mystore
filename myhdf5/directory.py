import os
import tempfile
import uuid
from typing import Iterable


class Dir:
    def __enter__(self):
        raise NotImplementedError()

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, *args, **kwargs):
        raise NotImplementedError()

    def __next__(self, suffix="") -> str:
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    @property
    def dirname(self) -> str:
        raise NotImplementedError()

    @property
    def names(self) -> Iterable[str]:
        raise NotImplementedError()


class InfinityTempNames(Dir):
    def __init__(self):
        self._published: set
        self._tmpdir: str
        self._dirname: str

    def __enter__(self):
        self._published = set()
        self._tmpdir = tempfile.TemporaryDirectory()
        self._dirname = self._tmpdir.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        tmpdir = self._tmpdir
        tmpdir.__exit__(*args, **kwargs)
        del self._tmpdir
        del self._published
        del self._dirname

    def __next__(self, suffix="") -> str:
        suffix = suffix or ""
        while True:
            filename = self._dirname + "/" + str(uuid.uuid4()) + suffix
            if (not os.path.exists(filename)) and filename not in self._published:
                break

        self._published.add(filename)
        return filename

    def __iter__(self):
        while True:
            yield self.__next__()

    def next(self, suffix=""):
        return self.__next__(suffix)

    @property
    def dirname(self):
        return self._dirname

    @property
    def names(self):
        return self._published.__iter__()


class RealDir(InfinityTempNames):
    def __init__(self, dirname, ignore_exists: bool = False):
        self._tmpdirname = dirname
        self._ignore_exists = ignore_exists

        if not ignore_exists and os.path.exists(dirname):
            raise FileNotFoundError()

        if not os.path.isdir(dirname):
            raise RuntimeError("Must be directory.")

    def __enter__(self):
        self._dirname = self._tmpdirname
        self._published = set()
        return self

    def __exit__(self, *args, **kwargs):
        del self._published
        del self._dirname
