from src.utils import decode


def test_keep():
    src_tokens = "This is sample example .".split()
    trg_tokens = src_tokens.copy()
    labels = ["$KEEP"]*len(src_tokens)
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_delete():
    src_tokens = "This is a huge sentence . We will need to delete this .".split()
    trg_tokens = "This is a sentence . We will need this .".split()
    labels = ["$KEEP", "$KEEP", "$KEEP", "$DELETE", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$DELETE", "$DELETE", "$KEEP", "$KEEP"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_append():
    src_tokens = "This is sentence . Where are you".split()
    trg_tokens = "This is a sentence . Where are you ?".split()
    labels = ["$KEEP", "$APPEND_a", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$APPEND_?"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_replace():
    src_tokens = "This is an sentence . Where are you .".split()
    trg_tokens = "This is a sentence . Where are you ?".split()
    labels = ["$KEEP", "$KEEP", "$REPLACE_a", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$REPLACE_?"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_merge_hyphen():
    src_tokens = "Tigers are cold blooded animals .".split()
    trg_tokens = "Tigers are cold-blooded animals .".split()
    labels = ["$KEEP", "$KEEP", "$MERGE_HYPHEN", "$KEEP", "$KEEP", "$KEEP"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_merge_space():
    src_tokens = "Water is every where .".split()
    trg_tokens = "Water is everywhere .".split()
    labels = ["$KEEP", "$KEEP", "$MERGE_SPACE", "$KEEP", "$KEEP"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_split_hyphen():
    src_tokens = "Tigers are cold-blooded animals .".split()
    trg_tokens = "Tigers are cold blooded animals .".split()
    labels = ["$KEEP", "$KEEP", "$TRANSFORM_SPLIT_HYPHEN", "$KEEP", "$KEEP"]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_transform_case():
    src_tokens = "this is a us sentence . Pcs have TWO 9 Gb rams .".split()
    trg_tokens = "This is a US sentence . PCs have two 9 Gb RAMs .".split()
    labels = [
        "$TRANSFORM_CASE_CAPITAL", "$KEEP", "$KEEP", "$TRANSFORM_CASE_UPPER", "$KEEP", "$KEEP",
        "$TRANSFORM_CASE_CAPITAL_1", "$KEEP", "$TRANSFORM_CASE_LOWER", "$KEEP", "$KEEP", "$TRANSFORM_CASE_UPPER_-1", "$KEEP",
    ]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_transform_agreement():
    src_tokens = "These are sentence . Pcs have a 9 Gb RAMs .".split()
    trg_tokens = "These are sentences . Pcs have a 9 Gb RAM .".split()
    labels = [
        "$KEEP", "$KEEP", "$TRANSFORM_AGREEMENT_PLURAL", "$KEEP",
        "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$TRANSFORM_AGREEMENT_SINGULAR", "$KEEP",
    ]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens


def test_transform_verb():
    src_tokens = "He eat apple . They were play football . I fix it .".split()
    trg_tokens = "He eats apple . They were playing football . I fixed it .".split()
    labels = [
        "$KEEP", "$TRANSFORM_VERB_VB_VBZ", "$KEEP", "$KEEP",
        "$KEEP", "$KEEP", "$TRANSFORM_VERB_VB_VBG", "$KEEP", "$KEEP",
        "$KEEP", "$TRANSFORM_VERB_VB_VBD", "$KEEP", "$KEEP",
    ]
    output_tokens, _ = decode(src_tokens, labels)
    assert output_tokens == trg_tokens
