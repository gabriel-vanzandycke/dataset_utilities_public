import abc
import json
import logging
import os
import sys

from tqdm import tqdm
from mlworkflow import Dataset, SideRunner


def export_dataset(dataset: Dataset, prefix, keys=None):
    """ Export the dataset to disk
    """
    files = []
    items = []

    keys = keys or dataset.yield_keys()

    for key in keys:
        item = dataset.query_item(key)
        files = files + list(item.files)
        items.append(item.db_item)
    with open(prefix + 'dataset.json', 'w') as fd:
        json.dump(items, fd)

    with open(prefix + 'files.txt', 'w') as fd:
        files = map(lambda x:x+'\n', files)
        fd.writelines(files)

def import_dataset(dataset_type, filename, **dataset_config):
    class ImportedDataset(Dataset):
        def __init__(self, filename, dataset_type, **dataset_config):
            with open(filename, "r") as fd:
                self.cache = json.load(fd)
            self._lookuptable = {}
            self.dataset_type = dataset_type
            self.dataset_config = dataset_config # I know it's a little bit ugly… but I need to move on to other things
        def yield_keys(self):
            for db_item in self.cache:
                item = self.dataset_type.items_type(db_item, **self.dataset_config)
                self._lookuptable[item.key] = item
                yield item.key
        def query_item(self, key, cache=False):
            try:
                if cache:
                    return self._lookuptable[key]
                else:
                    return self._lookuptable.pop(key)
            except KeyError as e:
                if key in list(self.keys):
                    raise KeyError("Item from '{}' was already queried. " \
                        "Use the 'cache' attribute of 'query_item' if you " \
                        "need to query your items multiple times. Or use a" \
                        "CachedDataset".format(key)) from e
                raise KeyError("Key '{}' not found. Did you call yield_keys() method?".format(key)) from e
    return ImportedDataset(filename=filename, dataset_type=dataset_type, **dataset_config)

class GenericItem(metaclass=abc.ABCMeta):
    """ Python object describing dataset item.
        Attributes that import files (like images) should be set as lazy
        properties with the @lazyproperty decorator from mlworkflow
    """
    @abc.abstractproperty
    def key(self):
        """ Generates the key associated to the Item.
            Key should to be immutable (eg: NamedTuple).
        """
        raise NotImplementedError
    @property
    def db_item(self):
        """ Returns the db_item that creates the python object
        """
        raise NotImplementedError
    @abc.abstractproperty
    def files(self):
        """ List files stored on remote storage that belong to the object
        """
        raise NotImplementedError

