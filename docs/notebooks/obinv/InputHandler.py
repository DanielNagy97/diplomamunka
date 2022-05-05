

class InputHandler(object):
    def __init__(self):
        pass

    def remove_punctuations(self, input_text):
        """
        Removes the punctiations from the input text.
        :param input_text: The input string.
        :return: A string without punctiation characters.
        """
        punctuations = [".", ",", ":", ";", "!", "?"]

        result_text = input_text
        for punctuation in punctuations:
            result_text = result_text.replace(punctuation, "")
        return result_text

    def make_lowercase(self, input_text):
        """
        Makes text lowercase.
        :param input_text: The input string.
        :return: A string with only lowercase characters.
        """
        lowercase_text = input_text.lower()
        return lowercase_text

    def tokenize_text(self, input_text):
        """
        Splits the input text into words.
        :param input_text: The input string.
        :return: A list of tokens(words).
        """
        tokens = input_text.split()
        return tokens

    def serialize_percentage_tokens(self, tokens):
        """
        Serializes the percentage tokens.
        :param tokens: The list of tokens
        :return: A list of tokens.
        """
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
        """
        Calling the normalizing functions in the right order.
        :param input_text: The input string.
        :return: A list of words(tokens).
        """
        normalized_text = self.remove_punctuations(input_text)
        normalized_text = self.make_lowercase(normalized_text)
        tokens = self.tokenize_text(normalized_text)
        tokens = self.serialize_percentage_tokens(tokens)

        return tokens

    def __call__(self, input_text):
        return self.normalize_text(input_text)