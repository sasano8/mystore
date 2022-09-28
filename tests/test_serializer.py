import io
import os
from functools import partial
from typing import Type

import h5py
import numpy as np
import pytest

from myhdf5 import BaseSerializer, ModelFile
from myhdf5.exceptions import FileAlreadyClosedError
from myhdf5.serializers import (
    ByteSerializer,
    Hdf5Serializer,
    JsonSerializer,
    NdarraySerializer,
    WieghtsSerializer,
)

from .conftest import InfinityTempNames, tmp_files


@pytest.fixture
def hdf5(tmp_files: InfinityTempNames):
    with h5py.File(tmp_files.next(".hdf5"), "w") as h5:
        yield h5


@pytest.mark.parametrize(
    "val, expect",
    [
        (None, True),
        ("", True),
        (0, True),
        (1.1, True),
        (True, True),
        ([], True),
        ({}, True),
        # invalid types
        (b"", False),
        (np.array([]), False),
    ],
)
def test_json_types(val, expect):
    assert JsonSerializer.is_instance(val) == expect


@pytest.mark.parametrize(
    "val, expect",
    [
        (b"", True),
        (io.BytesIO(b""), True),
        (partial(open, "_tmp", "wb"), True),
        (partial(open, "_tmp", "w"), False),
    ],
)
def test_byte_type(val, expect):
    if isinstance(val, partial):
        val = val()

    try:
        assert ByteSerializer.is_instance(val) == expect
    finally:
        # cleanup _tmp
        if isinstance(val, io.BufferedWriter):
            try:
                val.close()
                os.remove(val.name)
            except FileNotFoundError as e:
                ...


@pytest.mark.parametrize(
    "val, exc",
    [
        (None, None),
        (1, None),
        (1.1, None),
        ("", None),
        (True, None),
        (False, None),
        ([], None),
        ({}, None),
        ([1], None),
        ({"name": "a"}, None),
        # Out of range float values are not JSON compliant json.dump(..., allow_nan=False)
        (float("nan"), Exception),
        (float("inf"), Exception),
        (float("inf"), Exception),
    ],
)
def test_json_data(tmp_files: InfinityTempNames, val, exc):
    fname = tmp_files.next(".hdf5")

    if exc:
        with pytest.raises(exc):
            ModelFile.save(fname, val)
        return
    else:
        ModelFile.save(fname, val)

    assert ModelFile.load(fname) == val


def test_byte_data(tmp_files: InfinityTempNames):
    def assert_bytes(filename, expect):
        if isinstance(expect, io.BytesIO):
            expect = expect.read()
            input_ = io.BytesIO(expect)
        elif isinstance(expect, io.BufferedReader):
            input_ = expect
            expect = expect.read()
            input_.seek(0)
        else:
            input_ = expect

        ModelFile.save(filename, input_)
        with ModelFile.load_with(filename) as val:
            actual = b"".join(val)

        assert actual == expect
        return True

    # test buffer reader
    f1 = tmp_files.next()
    with open(f1, "wb") as f:
        f.write(b"xxx")

    assert assert_bytes(tmp_files.next(), open(f1, "rb"))
    assert assert_bytes(tmp_files.next(), io.BytesIO(b"xxx"))
    assert assert_bytes(tmp_files.next(), b"xxx")


@pytest.mark.parametrize(
    "cls, val, expect",
    [
        (NdarraySerializer, None, False),
        (NdarraySerializer, "", False),
        (NdarraySerializer, 0, False),
        (NdarraySerializer, 1.1, False),
        (NdarraySerializer, True, False),
        (NdarraySerializer, [], False),
        (NdarraySerializer, {}, False),
        (NdarraySerializer, b"", False),
        (NdarraySerializer, np.array([]), True),
        (NdarraySerializer, np.array([1]), True),
    ],
)
def test_numpy_types(cls: Type[BaseSerializer], val, expect):
    assert cls.is_instance(val) == expect


def test_numpy_data(tmp_files: InfinityTempNames):
    f1 = tmp_files.next(".hdf5")
    expect = np.array([1, 3, 5])
    ModelFile.save(f1, expect)
    with ModelFile.load_with(f1) as actual:
        np.equal(actual, expect)


def test_load_meta(tmp_files: InfinityTempNames):
    f1 = tmp_files.next()
    ModelFile.save(f1, 0, meta=None)
    assert ModelFile.load_meta(f1) == {}

    f2 = tmp_files.next()
    ModelFile.save(f2, 0, meta={})
    assert ModelFile.load_meta(f2) == {}

    f3 = tmp_files.next()
    ModelFile.save(f3, 0, meta={"val": 1})
    assert ModelFile.load_meta(f3) == {"val": 1}

    with pytest.raises(Exception, match="meta must be dict"):
        f4 = tmp_files.next()
        ModelFile.save(f4, 0, meta=[])

    with pytest.raises(Exception, match="meta must be dict"):
        f5 = tmp_files.next()
        ModelFile.save(f5, 0, meta=1)


def test_load_info(tmp_files: InfinityTempNames):
    f1 = tmp_files.next()
    ModelFile.save(f1, 0, meta=None)
    info = ModelFile.load_info(f1)
    assert info


def test_overwrite(tmp_files: InfinityTempNames):
    f1 = tmp_files.next()
    ModelFile.save(f1, 0, meta={"val": 1})
    assert ModelFile.load_meta(f1) == {"val": 1}

    with pytest.raises(FileExistsError):
        ModelFile.save(f1, 0, meta=None)

    ModelFile.save(f1, 0, meta={"val": 2}, overwrite=True)
    assert ModelFile.load_meta(f1) == {"val": 2}


def test_model_binary(tmp_files: InfinityTempNames):
    import keras
    import tensorflow as tf
    from keras.engine.functional import Functional

    model_file = tmp_files.next(".hdf5")
    output_file = tmp_files.next(".hdf5")

    model: Functional = tf.keras.applications.MobileNetV2(
        (32, 32, 3), classes=10, weights=None
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(model_file, save_format="h5", overwrite=True)

    with open(model_file, "rb") as f:
        actual = f.read()

    ModelFile.save_file(output_file, model_file)
    with ModelFile.load_with(output_file) as val:
        expect = b"".join(val)

    assert actual == expect


def test_save_weights(tmp_files: InfinityTempNames):
    import tensorflow as tf
    from keras.engine.functional import Functional

    model: Functional = tf.keras.applications.MobileNetV2(
        (32, 32, 3), classes=10, weights=None
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    weights_1 = model.get_weights()

    f1 = tmp_files.next(".hdf5")
    ModelFile.save_weights(f1, weights_1)
    weights_2 = ModelFile.load(f1)

    with pytest.raises(FileAlreadyClosedError):
        list(weights_2)

    with ModelFile.load_with(f1) as weights_2:
        from itertools import zip_longest

        for row1, row2 in zip_longest(weights_1, weights_2):
            np.equal(row1, row2)
