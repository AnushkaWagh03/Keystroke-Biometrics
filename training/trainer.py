from dataset.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.model = None

    def train(self):
        print("Training has been started")
        print(f"Using dataset: {self.config['dataset']['name']}")
        print(f"Model selected: {self.config['model']['name']}")
        
        # Load dataset
        self.dataset = DatasetFactory.create_dataset(self.config)
        self.dataset.load_data()
        
        # Create model
        self.model = ModelFactory.create_model(self.config)
        self.model.train([])
        
        print("Training completed.\n")

    def evaluate(self):
        print("Evaluating model...")
        
        # Generating dummy scores and labels for demonstration
        import numpy as np
        labels = [1]*50 + [0]*50
        scores = np.random.uniform(0.6, 0.9, 50).tolist() + np.random.uniform(0.1, 0.4, 50).tolist()
        
        results = {
            "scores": scores,
            "labels": labels,
            "ANGA": 0.12,
            "ANIA": 0.09
        }
        
        print("Evaluation completed.\n")
        return results
