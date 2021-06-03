from fast_wkw import DatasetHandle
import numpy as np
import asyncio
from pathlib import Path


async def test():
    dataset = DatasetHandle(
        str(
            Path(
                "/Users/norman/scalableminds/webknossos/_binaryData/sample_organization/l4_sample/segmentation/1"
            )
        )
    )
    response = await dataset.read_block((3072, 3072, 512))
    array = np.frombuffer(response.buf, dtype=np.dtype(response.dtype)).reshape(
        (response.num_channels, 32, 32, 32), order="F"
    )
    print(array, np.isfortran(array))


print("Hello")
asyncio.get_event_loop().run_until_complete(test())
