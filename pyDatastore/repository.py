from typing import Dict, Tuple

from .backends.backend import DatasetInfo


class Repository:
    def __init__(self) -> None:
        self.datasets: Dict[Tuple[str, str], Tuple[str, DatasetInfo]] = {}

    def add_dataset(self, backend_name: str, dataset: DatasetInfo) -> None:
        self.datasets[(dataset.organization_name, dataset.dataset_name)] = (
            backend_name,
            dataset,
        )

    def get_dataset(
        self, organization_name: str, dataset_name: str
    ) -> Tuple[str, DatasetInfo]:
        return self.datasets[(organization_name, dataset_name)]
