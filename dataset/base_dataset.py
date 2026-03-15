class BaseKeystrokeDataset:
    """Base class for all datasets"""
    
    def __init__(self, config):
        self.config = config
        self.data = []
        
    def load_data(self):
        """Load raw dataset"""
        raise NotImplementedError("Subclasses must implement load_data()")
        
    def normalize_to_standard_format(self):
        """Convert to standard format"""
        raise NotImplementedError("Subclasses must implement normalize_to_standard_format()")