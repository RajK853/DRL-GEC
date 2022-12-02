import os
import argparse
from zipfile import ZipFile

from src.utils import load_text, write_text, load_labels, load_model, iterative_prediction, clean_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BENCHMARK = "bea"
LABEL_PATH = "data/vocabs/labels.txt"
DATA_PATH = "data/processed/bea2019st/ABCN.test.bea19.orig"


def generate_outputs(policy, label_vocab, sentences, output_path, max_iter):
    print(f"Number of benchmark sentences: {len(sentences)}")
    corrected_sentences = iterative_prediction(policy, label_vocab, sentences, num_iter=max_iter, insert_start=True)
    write_text(corrected_sentences, output_path)


def benchmark(model_path, sentences, label_vocab, max_iter=10, force=False):
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    print(f"\n# Evaluating the model: {model_name}")
    benchmark_dir = os.path.abspath(os.path.join(model_dir, BENCHMARK))
    output_path = os.path.join(benchmark_dir, model_name.replace(".pt", ".out"))
    os.makedirs(benchmark_dir, exist_ok=True)
    if force or not os.path.exists(output_path):  # Generate model outputs
        print("# Generating model outputs")
        policy = load_model(model_name="roberta-base", model_path=model_path, num_labels=len(label_vocab))
        policy.eval()
        generate_outputs(policy, label_vocab, sentences, output_path, max_iter)
        ZipFile(output_path.replace(".out", ".zip"), mode="w").write(output_path, arcname=os.path.basename(output_path))
    else:
        print(f"# Output file already exists at '{output_path}'!")


def main(model_dir, label_path=LABEL_PATH, data_path=DATA_PATH, max_iter=10, force=False):
    sentences = load_text(data_path)
    sentences = [clean_text(sent) for sent in sentences]
    label_vocab = load_labels(label_path, verbose=True)
    model_names = [filename for filename in os.listdir(model_dir) if filename.endswith(".pt")]
    if model_names:
        for model_name in model_names:
            model_path = os.path.join(model_dir, model_name)
            benchmark(model_path, sentences, label_vocab, max_iter=max_iter, force=force)
    else:
        print(f"# No PyTorch model found in the given directory; {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to directory with the trained models', required=True)
    parser.add_argument('--label_path', help='Path to the label vocabulary', default=LABEL_PATH)
    parser.add_argument('--data_path', help='Path to the BEA-2019 test data', default=DATA_PATH)
    parser.add_argument('--max_iter', type=int, help='Max number of prediction iteration', default=10)
    parser.add_argument('--force', action="store_true")
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
