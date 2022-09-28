import numpy as np

from myhdf5 import ModelFile, TempModelStore
from myhdf5.aggregate import aggregate


def test_aggregate():
    import tensorflow as tf
    from keras.engine.functional import Functional

    model: Functional = tf.keras.applications.MobileNetV2(
        (32, 32, 3), classes=10, weights=None
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    with TempModelStore() as store:
        weights_1 = model.get_weights()
        f1 = store.file()
        f1.save_weights(weights_1, meta={"sample_num": 50})

        f2 = store.file()
        f2.save_weights(weights_1, meta={"sample_num": 80})

        def iterate(*args: ModelFile):
            for file in args:
                weights = file.load(map=list)
                sample_num = file.load_meta()["sample_num"]
                yield weights, sample_num

        results = aggregate(list(iterate(f1, f2)))
        assert results
