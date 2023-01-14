import re
import os
import argparse
from tqdm.auto import tqdm
from autocorrect import Speller
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from cdifflib import CSequenceMatcher as SequenceMatcher

from src.utils import write_json, clean_text

TITLE = """
######################
# Process M2 to JSON #
######################
"""
ELLIPSIS_PATTERN = r"(\.\s){2,}"          # Two or more sequence of ". "
PARENTHESIS_PATTERN = r"\((.*?)\)(\s)*"   # From "This is ok ( I guess ) ." match "( I guess ) "
Edit = Tuple[int, int, str]


def m2_parser(data_path: str) -> Tuple[str, Dict[int, List[Edit]]]:
    """
    Extract sentence and annotator edits from the M2 file
    """
    orig_sent = None
    annotator_edits = defaultdict(list)
    with open(data_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("S "):
                orig_sent = line[2:]
            elif line.startswith("A "):
                assert orig_sent is not None
                start_end, edit_type, edit_span, _, _, annotator_id = line[2:].split("|||")
                if edit_type != "noop":
                    start, end = start_end.split()
                    edit = (int(start), int(end), edit_span)
                    annotator_edits[annotator_id].append(edit)
            else:
                if orig_sent is not None:
                    yield orig_sent, annotator_edits
                orig_sent = None
                annotator_edits = defaultdict(list)


def filter_duplicate(annot_edits: Dict[int, List[Edit]]) -> List[List[Edit]]:
    """
    Remove duplicate annotator edits
    """
    all_edits = []
    for edits in annot_edits.values():
        if edits not in all_edits:
            all_edits.append(edits)
    return all_edits


def apply(tokens: List[str], edits: List[Edit]) -> List[str]:
    """
    Apply the annotator edits to the source tokens to generate new tokens
    """
    tokens = tokens.copy()
    for edit in reversed(edits):
        i, j, span = edit
        if span == "":
            tokens[i:j] = []
        else:
            tokens[i:j] = [span]
    return tokens


def gen_references(text: str, all_edits: List[List[Edit]]) -> List[str]:
    """
    Generate reference sentences from given text and edits
    """
    if all_edits:
        tokens = text.split()
        references = []
        for edits in all_edits:
            ref_tokens = apply(tokens, edits)
            ref_text = clean_text(" ".join(ref_tokens), is_ref=True)
            ref_text = remove_parenthetical_text(ref_text)
            references.append(ref_text)
    else:
        references = [text]
    return references


def similar_ratio(text_a: str, text_b: str) -> float:
    """
    Calculate token-based similarity between two texts
    """
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    return SequenceMatcher(None, tokens_a, tokens_b).ratio()


def check_proper_sent(text: str) -> bool:
    """
    Check that the sentence starts and ends properly
    """
    tokens = text.split()
    if not tokens:
        return False
    # First token is Capitalized and the ending character of last token is one of the given characters
    return tokens[0].istitle() and tokens[-1][-1] in '.!?"'


def check_ellipsis(text: str) -> bool:
    """
    Check if the sentence has any ellipsis
    """
    return bool(re.search(ELLIPSIS_PATTERN, text))


def correct_spelling(checker: Speller, text: str) -> str:
    """
    Correct any typos from the sentence
    """
    return checker(text)


def remove_parenthetical_text(text: str) -> str:
    """
    Remove parenthetical texts i.e. from "I am fine (or not)." to "I am fine."
    """
    return re.sub(PARENTHESIS_PATTERN, "", text)


def process_sent(
        text: str,
        annot_edits: Dict[int, List[Edit]],
        checker: Speller,
        min_len: int,
        max_len: int,
        min_sim: float,
        only_proper_sent: bool,
        spell_check: bool = False,
) -> Union[str, Dict[str, str]]:
    """
    Process a given sentence
    """
    # Filter sentence with ellipsis
    if check_ellipsis(text):
        return "Ellipsis"
    text = clean_text(text, is_ref=False)
    text = remove_parenthetical_text(text)
    # Filter sentence based on number of tokens
    num_tokens = len(text.split())
    if num_tokens < min_len:
        return "Less Tokens"
    elif num_tokens > max_len:
        return "More Tokens"
    if spell_check:
        text = correct_spelling(checker, text)
    all_edits = filter_duplicate(annot_edits)
    references = gen_references(text, all_edits)
    # Filter sentence based on whether any of the references is not a proper sentence
    if only_proper_sent and any(not check_proper_sent(ref_sent) for ref_sent in references):
        return "Improper Sentence"
    if all_edits:
        # Filter sentence based on the mean similarity between the original and reference sentences
        mean_sim = sum(similar_ratio(text, ref_sent) for ref_sent in references) / len(references)
        if mean_sim < min_sim:
            return "Source-Reference Similarity"
    return {"text": text, "references": references}


def main(
        m2_path: str,
        json_path: str,
        min_len: int = 5,
        max_len: int = 50,
        min_sim: float = 0.8,
        only_proper_sent: bool = True,
        spell_check: bool = True,
        remove_ellipsis: bool = True,
):
    print(TITLE)
    assert json_path.lower().endswith(".json"), f"Not a JSON file; got '{json_path}'"
    json_data = []
    stats = defaultdict(int)
    checker = Speller(lang="en", fast=False, threshold=0)
    for orig_sent, annot_edits in tqdm(m2_parser(m2_path), desc="Processing"):
        result = process_sent(orig_sent, annot_edits, checker, min_len, max_len, min_sim, only_proper_sent, spell_check)
        if isinstance(result, dict):
            json_data.append(result)
        else:
            stats[result] += 1
    print(f"Number of sentences: {len(json_data)}")
    print("Report of filtered sentences.")
    for key, value in stats.items():
        print(f"{key:>30}: {value}")
    json_dir = os.path.dirname(json_path)
    filename = os.path.basename(json_path)[:-5]
    os.makedirs(json_dir, exist_ok=True)
    params = {
        "min_len": min_len,
        "max_len": max_len,
        "min_sim": min_sim,
        "spell_check": spell_check,
        "remove_ellipsis": remove_ellipsis,
        "only_proper_sent": only_proper_sent,
    }
    write_json(os.path.join(json_dir, f"{filename}_params.json"), params)
    write_json(os.path.join(json_dir, f"{filename}_metadata.json"), stats)
    write_json(json_path, json_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m2_path', help='Path to the input M2 file', required=True)
    parser.add_argument('--json_path', help='Path to the output JSON files', required=True)
    parser.add_argument('--min_len', type=int, help='Min number of tokens in original sentence', default=5)
    parser.add_argument('--max_len', type=int, help='Max number of tokens in original sentence', default=50)
    parser.add_argument('--min_sim', type=float, help='Min avg similarity between original and references', default=0.8)
    parser.add_argument('--only_proper_sent', help='Allow only proper reference sentences', action="store_true")
    parser.add_argument('--spell_check', help='Check spelling errors in original and references', action="store_true")
    parser.add_argument('--remove_ellipsis', help='Remove (source) sentences with ellipsis', action="store_true")
    parser.set_defaults(only_proper_sent=True, spell_check=True, remove_ellipsis=True)
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
