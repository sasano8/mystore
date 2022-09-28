from typing import TypeVar

import h5py

T = TypeVar("T", bound="BaseSerializer")


class BaseSerializer:
    # name
    priority = 0
    track_order = False

    @property
    def get_name(self):
        return self.name  # type: ignore

    @property
    def get_priority(self):
        return self.priority

    @classmethod
    def is_type(cls, type_name):
        return type_name == cls.name  # type: ignore

    @staticmethod
    def is_instance(obj) -> bool:
        raise NotImplementedError()

    @staticmethod
    def serialize(grp: h5py.Group, obj):
        raise NotImplementedError()

    @staticmethod
    def deserialize(grp: h5py.Group):
        raise NotImplementedError()

    @staticmethod
    def is_only_opend() -> bool:
        return False
