

class InputHandler(object):
    def __init__(self):
        pass

    def remove_puntuations(self, input_text):
        punctuations = [".", ",", ":", ";", "!", "?"]

        result_text = input_text
        for punctuation in punctuations:
            result_text = result_text.replace(punctuation, "")
        return result_text

    def make_lowercase(self, input_text):
        lowercase_text = input_text.lower()
        return lowercase_text

    def tokenize_text(self, input_text):
        tokens = input_text.split()
        return tokens

    def serialize_percentage_tokens(self, tokens):
        serialized_tokens = tokens.copy()
        percentage_symbols = ["%", "százalék"]
        unused_tokens = []
        for i in range(len(serialized_tokens)):
            for percentage_symbol in percentage_symbols:
                if percentage_symbol == serialized_tokens[i]:
                    if(serialized_tokens[i-1].isnumeric()):
                        serialized_tokens[i-1] = serialized_tokens[i-1]+"%"
                    unused_tokens.append(serialized_tokens[i])

        for i in unused_tokens:
            serialized_tokens.remove(i)

        return serialized_tokens

    def normalize_text(self, input_text):
        normalized_text = self.remove_puntuations(input_text)
        normalized_text = self.make_lowercase(normalized_text)
        tokens = self.tokenize_text(normalized_text)
        tokens = self.serialize_percentage_tokens(tokens)

        return tokens

    def __call__(self, input_text):
        return self.normalize_text(input_text)