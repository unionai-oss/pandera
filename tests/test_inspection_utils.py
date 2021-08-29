# pylint: disable=missing-function-docstring,missing-module-docstring
# pylint: disable=missing-class-docstring,bad-mcs-classmethod-argument
from pandera.inspection_utils import (
    is_classmethod_from_meta,
    is_decorated_classmethod,
)


class SomeMeta(type):
    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, **kwargs)

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def regular_method_meta(cls):
        return cls

    @classmethod
    def class_method_meta(mcs):
        return mcs

    @staticmethod
    def static_method_meta():
        return 1


class SomeClass(metaclass=SomeMeta):
    def regular_method(self):
        return self

    @classmethod
    def class_method(cls):
        return cls

    @staticmethod
    def static_method():
        return 2


class SomeChild(SomeClass):
    def regular_method_child(self):
        return self

    @classmethod
    def class_method_child(cls):
        return cls

    @staticmethod
    def static_method_child():
        return 3


def test_is_decorated_classmethod() -> None:
    some_instance = SomeClass()
    some_child = SomeChild()

    cls_methods_with_deco = {
        SomeMeta.class_method_meta,
        SomeClass.class_method_meta,
        SomeClass.class_method,
        SomeChild.class_method_meta,
        SomeChild.class_method,
        SomeChild.class_method_child,
    }

    cls_methods_from_meta = {
        SomeClass.regular_method_meta,
        SomeChild.regular_method_meta,
    }

    all_methods = {
        # from meta
        SomeMeta.class_method_meta,
        SomeMeta.static_method_meta,
        # from parent
        SomeClass.class_method_meta,
        SomeClass.regular_method_meta,
        SomeClass.static_method_meta,
        SomeClass.class_method,
        some_instance.regular_method,
        SomeClass.static_method,
        # from child
        SomeChild.class_method_meta,
        SomeChild.regular_method_meta,
        SomeChild.static_method_meta,
        SomeChild.class_method,
        some_child.regular_method,
        SomeChild.static_method,
        SomeChild.class_method_child,
        some_child.regular_method_child,
        SomeChild.static_method_child,
    }

    for method in cls_methods_with_deco:
        assert is_decorated_classmethod(method), f"{method} is decorated"
    for method in all_methods - cls_methods_with_deco:
        assert not is_decorated_classmethod(
            method
        ), f"{method} is not decorated"
    for method in cls_methods_from_meta:
        assert is_classmethod_from_meta(method), f"{method} comes from meta"
    for method in all_methods - cls_methods_from_meta:
        assert not is_classmethod_from_meta(
            method
        ), f"{method} does not come from meta"
