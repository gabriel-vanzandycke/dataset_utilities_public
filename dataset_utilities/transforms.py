from abc import ABCMeta, abstractmethod
import random
from dataset_utilities.utils import jpegBlur

class Transform(metaclass=ABCMeta):
    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __gt__(self, other):
        return self.__repr__() > other.__repr__()

    @abstractmethod
    def __call__(self, key, item):
        pass

    def __repr__(self):
        config = getattr(self, "config", {k:v for k, v in self.__dict__.items() if not k.startswith("_")})
        attributes = ",".join("{}={}".format(k, v) for k,v in config.items())
        return "{}({})".format(self.__class__.__name__, attributes)


class DoNothing(Transform):
    def __call__(self, key, item):
        return item


class DeclutterItems(Transform):
    """ Drops attributes from dataset items. Attributes to drop are given by the
        'drop' argument.
    """
    def __init__(self, drop):
        self.drop = drop
    def __call__(self, key, item):
        for name in self.drop:
            delattr(item, name)
        return item


class JPEGCompressionTransform(Transform):
    def __init__(self, key_names, q_range=(30,60)):
        self.key_names = key_names
        self.q_range = q_range
        assert len(q_range) == 2 and q_range[0] < q_range[1] and q_range[0] > 0 and q_range[1] <= 100

    def __call__(self, key, data):
        if data is None:
            return None
        q = random.randint(*self.q_range)
        for k in data:
            if k in self.key_names:
                data[k] = jpegBlur(data[k], q)
        return data
