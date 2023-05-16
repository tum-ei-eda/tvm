import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
# TODO: --log
parser.add_argument("--output-dir", default="gen", help="TODO")
parser.add_argument("--target-name", default="riscv", help="TODO")
parser.add_argument("--hw-props", default=None, help="TODO")
parser.add_argument("--isa-props", default=None, help="TODO")
parser.add_argument("--intrin-props", default=None, help="TODO")
parser.add_argument("--uma-props", default=None, help="TODO")
parser.add_argument("--workload-props", default=None, help="TODO")
parser.add_argument("--deployment-props", default=None, help="TODO")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")
