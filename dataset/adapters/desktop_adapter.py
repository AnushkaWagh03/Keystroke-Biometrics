from dataset.base_dataset import BaseKeystrokeDataset

class DesktopDataset(BaseKeystrokeDataset):
    def __init__(self, config):
        super().__init__(config)
        
    def load_data(self):
        print("Loading Desktop dataset")
        return self
        
    def normalize_to_standard_format(self):
        print("Normalizing Desktop data")
        return self.data
