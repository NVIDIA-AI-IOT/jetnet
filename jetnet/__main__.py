from argparse import ArgumentParser
from asyncio import subprocess
from jetnet.build import (
    register_args as register_build_args,
    run_args as run_build_args
)
from jetnet.profile.profile import (
    register_args as register_profile_args,
    run_args as run_profile_args
)
from jetnet.demo.demo import (
    register_args as register_demo_args,
    run_args as run_demo_args
)


def register_args(parser):
    subparsers = parser.add_subparsers()

    parser_build = subparsers.add_parser("build")
    register_build_args(parser_build)
    parser_build.set_defaults(func=run_build_args)

    parser_profile = subparsers.add_parser("profile")
    register_profile_args(parser_profile)
    parser_profile.set_defaults(func=run_profile_args)

    parser_demo = subparsers.add_parser("demo")
    register_demo_args(parser_demo)
    parser_demo.set_defaults(func=run_demo_args)


def run_args(args):
    if args.func:
        args.func(args)


def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)


if __name__ == "__main__":
    main()