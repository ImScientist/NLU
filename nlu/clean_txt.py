import re
from .variables import puncts, mispell_dict


def remove_common_misspellings(text):
    for k, v in mispell_dict.items():
        text = re.sub(pattern=k, repl=v, string=text)

    return text


def remove_common_misspellings_v2(text):
    for k, v in mispell_dict.items():
        text = re.sub(pattern=k, repl=v, string=text)

    for el in puncts:
        text = text.replace(el, ' ' + '#' * len(el))

    return text
