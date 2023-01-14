import os
import argparse
import subprocess as sb

from src.utils import load_labels, load_model, read_m2, write_text, iterative_prediction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TITLE = """
##############
# CoNLL-2014 #
##############
"""
BENCHMARK = "conll"
LABEL_PATH = "data/vocabs/labels.txt"
SUBMODULE_PATH = os.path.abspath("m2scorer/")
BENCHMARK_DATA_PATH = os.path.join(SUBMODULE_PATH, f"conll14st-test-data/alt/official-2014.combined-withalt.m2")


def generate_outputs(policy, label_vocab, m2_path, output_path, max_iter):
    sentences = read_m2(m2_path)
    print(f"# Number of benchmark sentences: {len(sentences)}")
    corrected_sentences = iterative_prediction(policy, label_vocab, sentences, num_iter=max_iter, insert_start=True)
    write_text(corrected_sentences, output_path)


def benchmark(model_path, label_path=LABEL_PATH, m2_path=BENCHMARK_DATA_PATH, max_iter=10, force=False):
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    print(f"\n# Evaluating the model: {model_name}")
    benchmark_dir = os.path.abspath(os.path.join(model_dir, BENCHMARK))
    output_path = os.path.join(benchmark_dir, model_name.replace(".pt", ".out"))
    score_path = os.path.join(benchmark_dir, model_name.replace(".pt", ".score"))
    os.makedirs(benchmark_dir, exist_ok=True)
    if force or not os.path.exists(output_path):  # Generate model outputs
        print("# Generating model outputs")
        label_vocab = load_labels(label_path, verbose=True)
        policy = load_model(model_name="roberta-base", model_path=model_path, num_labels=len(label_vocab))
        policy.eval()
        generate_outputs(policy, label_vocab, m2_path, output_path, max_iter=max_iter)
    else:
        print(f"# Output file already exists at '{output_path}'!")
    # Execute the M2Scorer
    with open(score_path, "w") as fp:
        cmd = ["./m2scorer", "-v", output_path, m2_path]
        sb.Popen(cmd, cwd=SUBMODULE_PATH, stdout=fp)
    print(f"# Model score saved to '{score_path}'!")


def main(model_dir, label_path=LABEL_PATH, m2_path=BENCHMARK_DATA_PATH, max_iter=10, force=False):
    m2_path = os.path.abspath(m2_path)
    model_dir = os.path.abspath(model_dir)
    model_names = [filename for filename in os.listdir(model_dir) if filename.endswith(".pt")]
    print(TITLE)
    if model_names:
        for model_name in model_names:
            model_path = os.path.join(model_dir, model_name)
            benchmark(model_path, label_path=label_path, m2_path=m2_path, max_iter=max_iter, force=force)
    else:
        print(f"# No PyTorch model found in the given directory; {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to directory with the trained models', required=True)
    parser.add_argument('--label_path', help='Path to the label vocabulary', default=LABEL_PATH)
    parser.add_argument('--m2_path', help='Path to the benchmark M2 file', default=BENCHMARK_DATA_PATH)
    parser.add_argument('--max_iter', type=int, help='Max number of prediction iteration', default=10)
    parser.add_argument('--force', action="store_true")
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
