from dataset.base_dataset import BaseKeystrokeDataset

class CMUDataset(BaseKeystrokeDataset):
    def __init__(self, config):
        super().__init__(config)
        
    def load_data(self):
        print("Loading CMU dataset")
        return self
        
    def normalize_to_standard_format(self):
        print("Normalizing CMU data")
        return self.data
