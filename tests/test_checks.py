from autocorrect import Speller

from m2_to_json import check_proper_sent, check_ellipsis, correct_spelling


def test_proper_sent():
    proper_sents = (
        "This is fine .",
        "How are you ?",
        "Congratulations !",
        'He asked , " how are you ? "',
    )
    improper_sents = (
        "this is not fine .",
        "How are you",
        "congratulations",
        'he asked , " how are you ? "',
    )
    assert all(check_proper_sent(sent) for sent in proper_sents)
    assert all(not check_proper_sent(sent) for sent in improper_sents)


def test_ellipsis():
    ellipsis_sents = (
        "Jealousy . . . is a mental cancer .",
        "Yeah ? Well , you can just . . . .",
    )
    non_ellipsis_sents = (
        "This is fine .",
        "How are you ?",
    )
    assert all(check_ellipsis(sent) for sent in ellipsis_sents)
    assert all(not check_ellipsis(sent) for sent in non_ellipsis_sents)


def test_spell_correction():
    typo_correct_pairs = (
        #   Sentence with typo           Corrected Sentence
        ("This is not acceptible .", "This is not acceptable ."),
        ("I do not have that in calender .", "I do not have that in calendar ."),
        ("Aggreement is not possible .", "Agreement is not possible ."),
    )
    checker = Speller(lang="en", fast=False, threshold=0)
    assert all(checker(typo) == correct for (typo, correct) in typo_correct_pairs)
