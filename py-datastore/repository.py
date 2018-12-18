class Repository:

  def __init__(self):
    self.datasets = {}


  def add_dataset(self, backend_name, dataset):
    self.datasets[(dataset.organization_name, dataset.dataset_name)] = (backend_name, dataset)


  def get_dataset(self, organization_name, dataset_name):
    return self.datasets[(organization_name, dataset_name)]
