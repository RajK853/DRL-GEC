import os
import sys
import argparse
import subprocess as sb

from src.utils import load_labels, load_model, load_text, write_text, iterative_prediction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BENCHMARK = "jfleg"
LABEL_PATH = "data/vocabs/labels.txt"
SUBMODULE_PATH = os.path.abspath("jfleg/")


def generate_outputs(policy, label_vocab, data_path, output_path, max_iter):
    sentences = load_text(data_path)
    print(f"Number of benchmark sentences: {len(sentences)}")
    corrected_sentences = iterative_prediction(policy, label_vocab, sentences, num_iter=max_iter, insert_start=True)
    write_text(corrected_sentences, output_path)


def benchmark(model_path, label_path=LABEL_PATH, repo_path=SUBMODULE_PATH, max_iter=10, force=False):
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    print(f"\n# Evaluating the model: {model_name}")
    benchmark_dir = os.path.abspath(os.path.join(model_dir, BENCHMARK))
    os.makedirs(benchmark_dir, exist_ok=True)
    for benchmark_type in ("dev", "test"):
        print(f"# JFLEG Dataset: {benchmark_type.title()}")
        data_path = os.path.join(repo_path, f"{benchmark_type}/{benchmark_type}.spellchecked.src")
        output_path = os.path.join(benchmark_dir, model_name.replace(".pt", f"_{benchmark_type}.out"))
        score_path = os.path.join(benchmark_dir, model_name.replace(".pt", f"_{benchmark_type}.score"))
        if force or not os.path.exists(output_path):       # Generate model outputs
            print("# Generating model outputs")
            label_vocab = load_labels(label_path, verbose=True)
            policy = load_model(model_name="roberta-base", model_path=model_path, num_labels=len(label_vocab))
            policy.eval()
            generate_outputs(policy, label_vocab, data_path, output_path, max_iter=max_iter)
        else:
            print(f"# Output file already exists at '{output_path}'!")
        # Execute the JFLEG script
        src_path = os.path.join(repo_path, f"{benchmark_type}/{benchmark_type}.src")
        ref_paths = [f"{benchmark_type}/{benchmark_type}.ref{i}" for i in range(4)]
        with open(score_path, "w") as fp:
            cmd = [sys.executable, "./eval/gleu.py", "--ref", *ref_paths, "--src", src_path, "--hyp", output_path]
            sb.Popen(cmd, cwd=SUBMODULE_PATH, stdout=fp)
        print(f"# Model score saved to '{score_path}'!")
    print()


def main(model_dir, label_path=LABEL_PATH, repo_path=SUBMODULE_PATH, max_iter=10, force=False):
    repo_path = os.path.abspath(repo_path)
    model_dir = os.path.abspath(model_dir)
    model_names = [filename for filename in os.listdir(model_dir) if filename.endswith(".pt")]
    print(f"# Benchmarking with JFLEG dataset")
    if model_names:
        for model_name in model_names:
            model_path = os.path.join(model_dir, model_name)
            benchmark(model_path, label_path=label_path, repo_path=repo_path, max_iter=max_iter, force=force)
    else:
        print(f"# No PyTorch model found in the given directory; {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to directory with the trained models', required=True)
    parser.add_argument('--label_path', help='Path to the label vocabulary', default=LABEL_PATH)
    parser.add_argument('--repo_path', help='Path to the JFLEG repo', default=SUBMODULE_PATH)
    parser.add_argument('--max_iter', type=int, help='Max number of prediction iteration', default=10)
    parser.add_argument('--force', action="store_true")
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
