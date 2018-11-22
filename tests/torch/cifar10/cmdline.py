import argparse


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--outputdir", dest='outputdir')
    parser.add_argument(
        "--device",
        dest='device',
        default="cpu",
        choices=['cpu', 'ipu_model', 'sim', 'hw'])
    parser.add_argument('--hw_id', dest="hw_id", type=int)
    return parser.parse_args()
