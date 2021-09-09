### Dataset tooling library

The API builds on the MLWorkflow library, in which datasets are described by (key, item) pairs.
Among all the features of such datasets, the most useful are the `keys` attribute that allows browsing the
dataset keys in a lazy way and the `query_item(key)` method that outputs a dataset item given its key.

While the library original purpose is to be a uniform data loader, it also contains many dataset transformation tools.

### Existing datasets

Instants Dataset - a dataset of instants
Produced Sequences Dataset - a dataset of sequences
