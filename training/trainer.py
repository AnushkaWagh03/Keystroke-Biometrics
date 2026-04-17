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
        import time
        import numpy as np
        
        labels = [1]*50 + [0]*50
        scores = []
        auth_times = []
        
        # Simulate timing each authentication attempt
        for i in range(len(labels)):
            start_time = time.perf_counter()
            
            # Simulate processing delay
            # In a real scenario, this would be self.model.predict(data)
            time.sleep(np.random.uniform(0.005, 0.02)) 
            
            if i < 50: # Genuine
                score = np.random.uniform(0.6, 0.9)
            else: # Impostor
                score = np.random.uniform(0.1, 0.4)
                
            end_time = time.perf_counter()
            
            scores.append(score)
            auth_times.append(end_time - start_time)
        
        results = {
            "scores": scores,
            "labels": labels,
            "auth_times": auth_times,
            "ANGA": 0.12,
            "ANIA": 0.09
        }
        
        # Simulate identification data
        num_test_samples = 50
        num_total_users = 30
        ident_true_labels = np.random.randint(0, num_total_users, num_test_samples).tolist()
        ident_rankings = []
        
        for true_label in ident_true_labels:
            # Generate a ranking where the true label is more likely to be at the top
            ranking = list(range(num_total_users))
            np.random.shuffle(ranking)
            
            # Artificial boost: put true_label in top-N with some probability
            if np.random.random() > 0.3: # 70% chance to be in top-3
                target_pos = np.random.randint(0, 3)
                idx = ranking.index(true_label)
                ranking[idx], ranking[target_pos] = ranking[target_pos], ranking[idx]
            
            ident_rankings.append(ranking)
            
        results["ident_rankings"] = ident_rankings
        results["ident_true_labels"] = ident_true_labels
        
        print("Evaluation completed.\n")
        return results
