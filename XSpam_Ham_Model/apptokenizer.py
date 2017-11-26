import nltk


def tokenizer(text):
    """
    tokenizing the text as words
    :param text:
    :return:
    """
    words = nltk.word_tokenize(text)
    return words
