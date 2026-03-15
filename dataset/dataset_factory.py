from dataset.adapters.aalto_adapter import AaltoDataset
from dataset.adapters.cmu_adapter import CMUDataset
from dataset.adapters.desktop_adapter import DesktopDataset

class DatasetFactory:
    @staticmethod
    def create_dataset(config):
        name = config['dataset']['name'].lower()
        
        if name == 'aalto':
            return AaltoDataset(config)
        elif name == 'cmu':
            return CMUDataset(config)
        elif name == 'desktop':
            return DesktopDataset(config)
        elif name == 'dummy_dataset':
            return AaltoDataset(config)
        else:
            raise ValueError(f"Unknown dataset: {name}")
