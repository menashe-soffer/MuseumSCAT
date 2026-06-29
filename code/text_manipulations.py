


def postprocess(text):
    if not isinstance(text, str) or text.upper() == "MISSING":
        return "MISSING" if text.upper() == "MISSING" else text
    for old, new in {"ö": "ø", "Ö": "Ø", "ä": "æ", "Ä": "Æ", "ü": "y", "Ü": "Y", "ÿ": "y", "Ÿ": "Y"}.items():
        text = text.replace(old, new)

    return text



def atomize(text):
    # Split into words, then atomize each word, then join with triple spaces
    words = text.split(' ')
    atomized_words = [" ".join(list(word)) for word in words]
    return "   ".join(atomized_words)


def unatomize(atomized_text):

    words = atomized_text.split("   ")
    clean_words = [word.replace(" ", "") for word in words]
    return " ".join(clean_words)


