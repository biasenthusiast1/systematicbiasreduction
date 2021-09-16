"""
22.2.2021
Author Shirin
"""
import re


def preprocess_robust04(str_text):
    """
    extracts the bodey of the text
    between <text> and <\text>
    :param: str_text
    :return:
    """
    begin_idx = str_text.find("<TEXT>")
    end_idx = str_text.find("</TEXT>")
    return str_text[begin_idx+6:end_idx]


def preprocess_gov2(str_text):
    """

    :return:
    """
    # remove punctuations
    punc_list = [".", ",", "?", "!", "(", ")",
                 "-", "_", "*", "#", "@", "$",
                 "%", "^", "+", "...", "/", "{","}", "[", "]", "|" ]
    for punc in punc_list:
        str_text = str_text.replace(punc,"")
    # remove html tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str_text)


def preprocess_cw(str_text):
    """

    :param str_text:
    :return:
    """
    # remove html tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str_text)