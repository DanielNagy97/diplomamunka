

class TokenHandler(object):
    def __init__(self):
        pass

    def compare_two_tokens(self, token_a, token_b):
        best_match = 0.0
        offset = 0
        while(offset <= len(token_b) - len(token_a)):
            count = 0
            for i in range(len(token_a)):
                if(token_a[i] == token_b[offset + i]):
                    count += 1
                else:
                    break
            match_value = count/len(token_a)

            if(match_value > best_match):
                best_match = match_value
            offset += 1
        return best_match

    def find_best_match_for_token_among_tokens(self, test_token, tokens):
        best_match = 0.0
        for token in tokens:
            match_value = self.compare_two_tokens(token, test_token)
            if(match_value > best_match):
                best_match = match_value
        return best_match