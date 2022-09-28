from myhdf5 import TempModelStore


def test_temp_model_store():
    with TempModelStore() as store:
        f1 = store.file()
        f1.save({})
        assert f1.load() == {}

        f2 = store.file()
        f2.save({})
        assert f2.load() == {}

        f3 = store.file()
        f3.save({})
        assert f3.load() == {}

        result = store.list_by_updated_at()
        assert result
