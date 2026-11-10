import argparse
import json
from pathlib import Path

from intent_router.data import download
from intent_router.model import IntentRouter, train


def main():
    parser = argparse.ArgumentParser(prog="intent-router")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--data", type=Path, default=Path("data/raw"))
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("data/raw"))
    fit.add_argument("--output", type=Path, default=Path("artifacts"))
    predict = sub.add_parser("predict")
    predict.add_argument("text")
    predict.add_argument("--model", type=Path, default=Path("artifacts/router.joblib"))
    args = parser.parse_args()
    if args.command == "download":
        download(args.data)
        print("Pinned dataset downloaded and checksums verified.")
    elif args.command == "train":
        report = train(args.data, args.output)
        print(
            json.dumps(
                {
                    "model": report["selected_model"],
                    "accuracy": report["test"]["accuracy"],
                    "macro_f1": report["test"]["macro_f1"],
                    "coverage": report["test"]["coverage"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(IntentRouter.load(args.model).predict(args.text), indent=2))


if __name__ == "__main__":
    main()
