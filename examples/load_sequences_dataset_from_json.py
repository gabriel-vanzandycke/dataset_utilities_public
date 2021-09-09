from matplotlib import pyplot as plt

from dataset_utilities.ds.generic_dataset import ImportedDataset
from dataset_utilities.ds.produced_sequences_dataset import ProducedSequencesDataset, KeypointCorrectionTransform

from mlworkflow import TransformedDataset

ds = ImportedDataset("sequences_dataset.json", ProducedSequencesDataset, dataset_folder="/DATA/datasets/dza/Basketball_Player_Annotation")

# Transform the dataset to correct the NEWS keepoints
ds = TransformedDataset(ds, [KeypointCorrectionTransform()])

key = next(iter(ds.yield_keys()))
item = ds.query_item(key)

fig, axs = plt.subplots(2, 10)
for i, tba in enumerate(item.get_thumbnails(list(item.tracklets.keys())[0])):
    if i >= 10:
        axs[1, i-10].imshow(tba.image)
    else:
        axs[0, i].imshow(tba.image)
plt.savefig('tracklet_example.png')
