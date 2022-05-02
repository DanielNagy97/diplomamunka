from obinv.TokenHandler import TokenHandler

class StructuredDataMaker(object):
    def __init__(self, synonyms):
        self.synonyms = synonyms
        self.token_handler = TokenHandler()

    def find_classes(self, tokens, threshold):
        result_classes = []
        for i in range(len(tokens)):
            for key in self.synonyms.keys():
                match_value =\
                    self.token_handler.find_best_match_for_token_among_tokens(
                        tokens[i],
                        self.synonyms[key]
                    )
                if(match_value >= threshold):
                    found_class = {"token_id" : i, "class" : key}
                    if not found_class in result_classes:
                        result_classes.append(found_class)
        return result_classes

    def find_percentage_values(self, tokens):
        res = []
        for i in range(len(tokens)):
            if "%" in tokens[i]:
                prob = int(tokens[i].split("%")[0]) / 100
                found_class = {"token_id" : i, "value" : prob}
                if not found_class in res:
                    res.append(found_class)
        return res

    def make_dict_from_classes(self, res_classes, res_probs):
        class_names = self.synonyms.keys()
        base_data = dict()

        for class_entity in class_names:
            base_data[class_entity] = 0.0

        if(len(res_probs) > 0):
            found_classes = []
            sums_of_prod = 0
            for i in range(len(res_probs)):
                for j in range(len(res_classes)):
                    if(res_probs[i]["token_id"] < res_classes[j]["token_id"]):
                        base_data[res_classes[j]['class']] = res_probs[i]['value']
                        sums_of_prod += res_probs[i]['value']
                        found_classes.append(res_classes[j])
                        break
            if(sums_of_prod != 1):
                if(len(found_classes) < len(res_classes)):
                    for found_class in found_classes:
                        res_classes.remove(found_class)
                    for res_class in res_classes:
                        base_data[res_class['class']] = (1 - sums_of_prod) / len(res_classes)
                elif(len(found_classes) == len(res_classes)):
                    remaining_classes = list(class_names)
                    for found_class in found_classes:
                        remaining_classes.remove(found_class['class'])
                    for remaining_class in remaining_classes:
                        base_data[remaining_class] = (1 - sums_of_prod) / len(remaining_classes)
        
        else:
            for found_class in map(lambda x: x["class"], res_classes):
                base_data[found_class] = 1 / len(res_classes)
        return base_data


    def make_structured_data(self, tokens, threshold):
        result_classes = self.find_classes(tokens, threshold)
        result_percentages = self.find_percentage_values(tokens)
        
        return self.make_dict_from_classes(result_classes, result_percentages)

    def __call__(self, tokens, threshold):
        return self.make_structured_data(tokens, threshold)