import json
from mlworkflow import Dataset

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
                raise KeyError("Item from '{}' was already queried. Use the 'cache' attribute of 'query_item'" \
                    " if you need to query your items multiple times.".format(key)) from e
            raise KeyError("Key '{}' not found. Did you call yield_keys() method?".format(key)) from e
