class BaseKeystrokeModel:
    def __init__(self, config):
        self.config = config
        self.params = config['model'].get('parameters', {})
        
    def train(self, train_data):
        raise NotImplementedError
        
    def predict(self, sample, user_id):
        raise NotImplementedError
