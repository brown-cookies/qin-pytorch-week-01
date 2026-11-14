import argparse

MODEL_CHOICES = ("both", "manual", "auto")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m src.main")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "run",
        help="full demo: train, evaluate, plot"
    )
    p.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="both",
        help="which model to run (default: both)"
    )

    p = sub.add_parser(
        "split",
        help="write the train/validation/test"
    )
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "train",
        help="train the manual model and save a checkpoint"
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny, fast run for CI"
    )
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "evaluate",
        help="evaluate a checkpoint on the test set"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metrics-out", required=True)

    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)

    # Setting up the default args
    if args.command is None:
        args.command = "run"
        args.model = "both"

    return args
