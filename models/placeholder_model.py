from models.base_model import BaseKeystrokeModel

class PlaceholderModel(BaseKeystrokeModel):
    def __init__(self, config):
        super().__init__(config)
        print("PlaceholderModel initialized")
        
    def train(self, train_data):
        print("PlaceholderModel training completed")
        
    def predict(self, sample, user_id):
        return 0.5
