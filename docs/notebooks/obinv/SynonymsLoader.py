import json

class SynonymsLoader(object):
    def __init__(self):
        pass

    def load_synonyms(synonyms_path):
        with open(synonyms_path) as json_file:
            synonyms = json.load(json_file)
        return synonyms