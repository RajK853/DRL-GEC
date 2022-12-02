import argparse

import benchmark_bea
import benchmark_conll
import benchmark_jfleg


def main(model_dir, label_path, max_iter=10, force=False):
    benchmark_bea.main(model_dir, label_path, max_iter=max_iter, force=force)
    benchmark_conll.main(model_dir, label_path, max_iter=max_iter, force=force)
    benchmark_jfleg.main(model_dir, label_path, max_iter=max_iter, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to directory with the trained models', required=True)
    parser.add_argument('--label_path', help='Path to the label vocabulary', default=benchmark_conll.LABEL_PATH)
    parser.add_argument('--max_iter', type=int, help='Max number of prediction iteration', default=10)
    parser.add_argument('--force', action="store_true")
    # Convert parsed arguments into key-worded arguments
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
