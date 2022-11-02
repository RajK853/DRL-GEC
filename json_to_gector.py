import os
import argparse
from tqdm.auto import tqdm

from src.utils import load_json
from gector.utils import preprocess_data


def main(json_path: str, output_path: str = None, chunk_size: int = 1024):
    """
    Modified version of the GECToR's preprocessing script to generate their format of data from a JSON file instead
    of source and target files.
    Source:
    https://github.com/grammarly/gector/blob/3d41d2841512d2690cffce1b5ac6795fe9a0a5dd/utils/preprocess_data.py#L328
    """
    assert json_path.endswith(".json"), f"Not a JSON file; {json_path}"
    if output_path is None:
        output_path = json_path.replace(".json", ".gector")
        print(f"- Saving data to '{output_path}'")
    json_data = load_json(json_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    tagged = []
    cnt_total, cnt_all, cnt_tp = 0, 0, 0
    for example_dict in tqdm(json_data, desc="Preparing data", total=len(json_data), unit_scale=True):
        source_sent = example_dict["text"]
        target_sent = example_dict["references"][0]
        aligned_sent = preprocess_data.align_sequences(source_sent, target_sent)
        if source_sent != target_sent:
            cnt_tp += 1
        alignments = [aligned_sent]
        cnt_all += len(alignments)
        try:
            check_sent = preprocess_data.convert_tagged_line(aligned_sent)
        except Exception:
            # debug mode
            aligned_sent = preprocess_data.align_sequences(source_sent, target_sent)
            check_sent = preprocess_data.convert_tagged_line(aligned_sent)

        if "".join(check_sent.split()) != "".join(target_sent.split()):
            # do it again for debugging
            aligned_sent = preprocess_data.align_sequences(source_sent, target_sent)
            check_sent = preprocess_data.convert_tagged_line(aligned_sent)
            print(f"Incorrect pair: \n{target_sent}\n{check_sent}")
            continue
        if alignments:
            cnt_total += len(alignments)
            tagged.extend(alignments)

        if len(tagged) > chunk_size:
            preprocess_data.write_lines(output_path, tagged, mode="a")
            tagged = []

    if tagged:
        preprocess_data.write_lines(output_path, tagged, "a")

    print(f"Overall extracted: {cnt_total}\n"
          f"Sentences with grammar errors: {cnt_tp}\n"
          f"Sentences without grammar errors: {cnt_all - cnt_tp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="Path to the input JSON file", required=True)
    parser.add_argument("--output_path", help="Path to the output file", default=None)
    parser.add_argument("--chunk_size", help="Chunk size during processing", default=1024)
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
