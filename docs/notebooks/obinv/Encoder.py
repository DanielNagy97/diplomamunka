

class Encoder(object):
    """
    Encoder class
    """
    def __init__(self):
        pass

    def make_one_hot(self, input_object):
        """
        Makes an one-hot representation from the given object.
        :param input_object: The structured data object.
        :return: A list of floats.
        """
        one_hot = []
        for key in input_object.keys():
            one_hot.append(input_object[key])
        return one_hot