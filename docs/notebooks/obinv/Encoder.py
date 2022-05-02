

class Encoder(object):
    def __init__(self):
        pass

    def make_one_hot(self, input_object):
        one_hot = []
        for key in input_object.keys():
            one_hot.append(input_object[key])
        return one_hot