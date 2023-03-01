import os
import json
import tarfile
import pathlib
import tempfile
import shutil
import argparse
import numpy as np


from contextlib import contextmanager, nullcontext

import tvm
from tvm import relay
import tvm.contrib.utils
from tvm.contrib.download import download_testdata
from tvm import relay, transform

parser = argparse.ArgumentParser(description="TVM Script")

MODELS = ["aww", "vww", "resnet", "toycar"]

parser.add_argument("model", metavar="MODEL", type=str, choices=MODELS, help="The model name (choices: %(choices)s)")
parser.add_argument("--arch", type=str, default="rv32gcp", choices=["rv32gc", "rv32gcp"], help="The RISC-V architecture used by the compiler (default: %(default)s)")
parser.add_argument("--abi", type=str, default="ilp32d", choices=["ilp32", "ilp32d"], help="The RISC-V architecture used by the compiler (default: %(default)s)")
parser.add_argument("--device", type=str, default=None, choices=[None, "arm_cpu"], help="The used target device to determine the TVM schedules (default: %(default)s)")
parser.add_argument("--cpu", type=str, default=None, choices=[None, "cortex-m0", "cortex-m7"], help="Used to identify which CPU features are available (default: %(default)s)")
parser.add_argument("--data-layout", type=str, default="NHWC", choices=["NHWC", "NCHW"], help="Transform the data layout in the graph (optional)")
parser.add_argument(
    "--kernel-layout", type=str, default="default", choices=["default", "IOHW", "HWOI", "HWIO", "IHWO", "OHWI", "OIHW"], help="Transform the kernel layout in the graph (default: %(default)s)"
)
parser.add_argument("--verbose", action="store_true", help="Show all compilation outputs")
parser.add_argument("--disable-legalize", action="store_true", help="Force int8 data in conv2d and dense layers")
parser.add_argument(
    "--output", type=str, default="auto", help="Destination for tuning logs"
)
parser.add_argument("--append", action="store_true", help="Append existing tuning logs.")
parser.add_argument("--trials", type=int, default=100, help="Number of tuning iterations per task (default: %(default)s)")
parser.add_argument("--early-stopping", type=int, default=None, help="Stop tuning if performance doeas not impreove over a certain amount of iterations")


args = parser.parse_args()


model = args.model

if model == "aww":
    input_tensor = "input_1"
    input_shape = (1, 49, 10, 1)
    input_dtype = "int8"
elif model == "vww":
    input_tensor = "input_1_int8"
    input_shape = (1, 96, 96, 3)
    input_dtype = "int8"
elif model == "resnet":
    input_tensor = "input_1_int8"
    input_shape = (1, 32, 32, 3)
    input_dtype = "int8"
elif model == "toycar":
    input_tensor = "input_1"
    input_shape = (1, 640)
    input_dtype = "int8"
else:
    raise RuntimeError(f"Unsupported model: {model}")

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))
model_url = f"https://github.com/tum-ei-eda/mlonmcu-models/raw/main/{model}/{model}.tflite"
model_file = f"{model}.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Parse the python model object to convert it into a relay module
# and weights.

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

######################################################################
# Defining the target

RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})

target_str = "c"
if args.device:
    device = args.device
    target_str += f" -device={device}"
    cpu = None
    if args.cpu:
        cpu = args.cpu
    elif args.device == "arm_cpu":
        cpu = "cortex-m7"
    if cpu:
        target_str += f" -mcpu={cpu}"
else:
    device = "x86"
    cpu = "default"

TARGET = target_str


######################################################################
# Optionally convert graph layout

if args.data_layout:
    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": [args.data_layout, args.kernel_layout],
        "nn.conv2d_transpose": [args.data_layout, args.kernel_layout],
        "qnn.conv2d": [args.data_layout, args.kernel_layout],
    }

    # Convert the layout of the graph where possible.
    seq = transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
            relay.transform.FoldConstant(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        try:
            mod = seq(mod)
        except Exception as err:
            raise RuntimeError("Error converting layout to {0}: {1}".format(":".join([args.data_layout, args.kernel_layout]), str(err)))


######################################################################
# Now, compile the model for the target:

executor = relay.backend.Executor("graph", {"link-params": True})
# executor = relay.backend.Executor("graph", {"link-params": False})


@contextmanager
def OptionallyDisableLegalize(disableLegalize):
    if not disableLegalize:
        yield nullcontext()
        return
    from tvm.relay.testing.temp_op_attr import TempOpAttr

    def do_not_legalize(attrs, inputs, types):
        print("do_not_legalize")
        return None

    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", do_not_legalize) as denseCtx:
        with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", do_not_legalize) as convCtx:
            yield (denseCtx, convCtx)

if args.output == "auto":
    records_path = pathlib.Path.cwd() / "tuning_records" / args.model / f"{device}_{cpu}" / f"{args.data_layout}_{args.kernel_layout}"/ "spike" / f"{args.arch}_{args.abi}" / "tuning_records.log"
else:
    records_path = args.output
records_path.parent.mkdir(exist_ok=True, parents=True)

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=[]):
    with OptionallyDisableLegalize(args.disable_legalize):
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, TARGET)
assert len(tasks) > 0, "No tunable tasks found"

template_project_path = pathlib.Path.cwd() / "apps" / "microtvm" / "spike"

project_options = {
    "verbose": args.verbose,
    "spike_exe": os.getenv("SPIKE", None),
    "spike_pk": os.getenv("SPIKEPK", None),
    "arch": args.arch,
    "abi": args.abi,
    "triple": "riscv32-unknown-elf",
    "spike_extra_args": "",
    "pk_extra_args": "",
}

module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=pathlib.Path(template_project_path),
    project_options=project_options,
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=False,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

if not args.append:
    if os.path.exists(records_path):
        os.remove(records_path)

num_trials = args.trials
early_stopping = args.early_stopping
if not early_stopping:
    early_stopping = num_trials
elif early_stopping == "auto":
    early_stopping = max(10, num_trials // 2)
for task in tasks:
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file(str(records_path)),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
        early_stopping=early_stopping,
    )
