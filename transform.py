import numpy as np


class Transform:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def apply(self, image, anno, cls):
        raise NotImplementedError


class TransformFactory:
    @classmethod
    def get(cls, name, **kwargs):
        try:
            eval(name)
        except NameError:
            print(f"{name} not defined as a transformation, check your spelling")
            raise

        return eval(name)(**kwargs)

class Resize(Transform):
    def apply(self, image, anno, cls):
        raise NotImplementedError

