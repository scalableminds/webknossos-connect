class DatasetInfo:
    def to_webknossos(self):
        pass


class Backend:
    async def handle_new_dataset(self, organization_name, dataset_name, dataset_info):
        pass

    async def read_data(self, dataset, layer_name, resolution, offset, shape):
        pass
