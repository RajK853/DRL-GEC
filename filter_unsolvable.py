import os
import argparse

from src import utils

ROOT_PATH = os.path.dirname(__file__)
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")


def main(json_path: str, label_path: str, batch_size: int = 128):
    assert json_path.endswith(".json"), f"Not a JSON file. Got '{json_path}'"
    json_data = utils.load_json(json_path)
    label_vocab = utils.load_text(label_path)
    filtered_data = utils.filter_by_solvable(json_data, label_vocab, batch_size=batch_size)
    print(f"# Original Size: {len(json_data)}")
    print(f"# Size of filtered dataset: {len(filtered_data)}")
    output_json_path = json_path.replace(".json", "_filtered.json")
    utils.write_json(output_json_path, filtered_data)
    print(f"# Saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="Path to the input JSON file", type=str, required=True)
    parser.add_argument("--label_path", help="Path to the label vocabulary", type=str, default=DEFAULT_LABELS_PATH)
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
