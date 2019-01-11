from typing import Tuple


class DatasetInfo:
    organization_name: str
    dataset_name: str

    def to_webknossos(self):
        pass


class Backend:
    async def handle_new_dataset(self, organization_name, dataset_name, dataset_info):
        pass

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        resolution: int,
        offset: Tuple[int, int, int],
        shape: Tuple[int, int, int],
    ) -> None:
        pass
