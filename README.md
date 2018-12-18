# py-datastore
A webKnossos compatible datastore written in Python

## How to setup?
1. add datastore to wK database
2. adjust config in `py-datastore/__main__.py`
3. `./run`

## Development Status

The datastore currently has basic neuroglancer support.

working:
* health endpoint
* pinging wk to report health
* adding neuroglancer datasets
* reporting datasets to wk

not working:
* no proper error handling / status codes (in case of error, exceptions are raised)
* only single channel uint8 data (since wk backend / postgres schema does not support uint64 segmentation)
* does not handle different chunk_sizes well
* currently only considers the first `scale`/`resolution` (since wk client cannot handle all combinations of resolutions)
* thumbnail parameters are hard-coded
* datasets are not persisted
* quite slow, since no caching
* no authentication
