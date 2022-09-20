from argparse import ArgumentParser
from jetnet.utils import import_object


def register_args(parser):
    parser.add_argument("model_config", type=str)


def run_args(args):
    config = import_object(args.model_config).copy(deep=True)
    config.build()



def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)


if __name__ == "__main__":
    main()