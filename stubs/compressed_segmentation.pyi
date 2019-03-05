from typing import Tuple

import numpy as np

def decompress(
    encoded: bytes,
    volume_size: Tuple[int, int, int],
    dtype: str,
    block_size: Tuple[int, int, int],
    order: str,
) -> np.ndarray: ...
