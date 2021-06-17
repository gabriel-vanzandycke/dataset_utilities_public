from matplotlib import pyplot as plt

from dataset_utilities.ds.generic_dataset import import_dataset
from dataset_utilities.ds.instants_dataset import InstantsDataset

ds = import_dataset(InstantsDataset, "dataset.json", dataset_folder="/DATA/datasets/basketball-instants-dataset", download_flags=27)
key = next(iter(ds.yield_keys()))
item = ds.query_item(key)
plt.imshow(item.draw(1))
