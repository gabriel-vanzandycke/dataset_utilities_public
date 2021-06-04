from matplotlib import pyplot as plt

from dataset_utilities.ds.generic_dataset import ImportedDataset
from dataset_utilities.ds.instants_dataset import InstantsDataset

ds = ImportedDataset("dataset.json", InstantsDataset, dataset_folder="/DATA/datasets/basketball-instants-dataset", download_flags=11)
key = next(iter(ds.yield_keys()))
item = ds.query_item(key)
plt.imshow(item.draw(1))
