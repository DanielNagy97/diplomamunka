import json


class SynonymsLoader(object):
    """
    SynonymsLoader class
    """
    def __init__(self):
        pass

    def load_synonyms(synonyms_path):
        """
        Loads the synonyms from the json file into a dict.
        :param synonyms_path: The path of the synonyms file.
        :return: The synonyms dict object.
        """
        with open(synonyms_path) as json_file:
            synonyms = json.load(json_file)
        return synonyms