from tensorflow import keras

class ModelLoader(object):
    def __init__(self):
        pass

    def load_model(model_path):
        return keras.models.load_model(model_path)