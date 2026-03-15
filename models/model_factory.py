from models.placeholder_model import PlaceholderModel

class ModelFactory:
    @staticmethod
    def create_model(config):
        model_name = config['model']['name'].lower()
        
        if model_name == 'placeholder_model':
            return PlaceholderModel(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
