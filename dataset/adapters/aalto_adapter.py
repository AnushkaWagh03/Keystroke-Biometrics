from dataset.base_dataset import BaseKeystrokeDataset

class AaltoDataset(BaseKeystrokeDataset):
    def __init__(self, config):
        super().__init__(config)
        
    def load_data(self):
        print("Loading Aalto dataset")
        # Will implement actual loading later
        return self
        
    def normalize_to_standard_format(self):
        print("Normalizing Aalto data")
        return self.data
