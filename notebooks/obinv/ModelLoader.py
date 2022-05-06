from tensorflow import keras


class ModelLoader(object):
    """
    ModelLoader class
    """
    def __init__(self):
        pass

    def load_model(model_path):
        """
        Loads the given model.
        :param model_path: The path of the model.
        :return: A keras model.
        """
        return keras.models.load_model(model_path)