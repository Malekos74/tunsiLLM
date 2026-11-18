import argparse
from src.datasets.download import download_one
from src.datasets.inspect import main as inspect_main
from src.utils.config import load_dataset_config

def main():
    parser = argparse.ArgumentParser(description="Tunisian LLM dataset pipeline")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()
    configs = load_dataset_config()
    names = list(configs) if args.dataset == "all" else [args.dataset]
    for name in names:
        download_one(name, configs[name])

if __name__ == "__main__":
    main()
