import os
import pathlib
import shutil
import argparse


from contextlib import contextmanager, nullcontext

import tvm
from tvm import relay, transform
import tvm.contrib.utils
from tvm.contrib.download import download_testdata

parser = argparse.ArgumentParser(description="TVM Script")

MODELS = ["aww", "vww", "resnet", "toycar"]

parser.add_argument("model", metavar="MODEL", type=str, choices=MODELS, help="The model name (choices: %(choices)s)")
parser.add_argument("--arch", type=str, default="rv32gcp", choices=["rv32gc", "rv32gcp"], help="The RISC-V architecture used by the compiler (default: %(default)s)")
parser.add_argument("--abi", type=str, default="ilp32d", choices=["ilp32", "ilp32d"], help="The RISC-V architecture used by the compiler (default: %(default)s)")
parser.add_argument("--device", type=str, default=None, choices=[None, "arm_cpu"], help="The used target device to determine the TVM schedules (default: %(default)s)")
parser.add_argument("--cpu", type=str, default="cortex-m7", choices=[None, "cortex-m0", "cortex-m7"], help="Used to identify which CPU features are available (default: %(default)s)")
parser.add_argument("--data-layout", type=str, default="NHWC", choices=["NHWC", "NCHW"], help="Transform the data layout in the graph (optional)")
parser.add_argument(
    "--kernel-layout", type=str, default="default", choices=["default", "IOHW", "HWOI", "HWIO", "IHWO", "OHWI", "OIHW"], help="Transform the kernel layout in the graph (default: %(default)s)"
)
parser.add_argument("--verbose", action="store_true", help="Show all compilation outputs")
parser.add_argument("--profile", action="store_true", help="Profile the model execution layer by layer")
parser.add_argument("--disable-legalize", action="store_true", help="Force int8 data in conv2d and dense layers")
parser.add_argument(
    "--tuning-records", type=str, nargs="?", default=None, const="auto", help="Use tuning records for specified target, if available"
)


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

if args.tuning_records:
    if args.tuning_records == "auto":
        records_path = pathlib.Path.cwd() / "tuning_records" / args.model / f"{device}_{cpu}" / f"{args.data_layout}_{args.kernel_layout}" / "spike" / f"{args.arch}_{args.abi}" / "tuning_records.log"

    else:
        records_path = args.tuning_records
    assert os.path.exists(records_path), f"Tuning records file ({records_path}) does not exist!"
else:
    records_path = None

with tvm.autotvm.apply_history_best(records_path):
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=[]):
        with OptionallyDisableLegalize(args.disable_legalize):
            module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params, executor=executor)
            module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params, executor=executor)
# with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=[]):    module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)

######################################################################
# Inspecting the compilation output

c_source_module = module.get_lib().imported_modules[0]
c_source_code = c_source_module.get_source()
# print(c_source_code)


######################################################################
# Compiling the generated code

gen_path = pathlib.Path.cwd() / "gen"

if not gen_path.is_dir():
    gen_path.mkdir()

model_library_format_tar_path = gen_path / "mlf.tar"
tvm.micro.export_model_library_format(module, model_library_format_tar_path)

# TVM also provides a standard way for embedded platforms to automatically generate a standalone
# project, compile and flash it to a target, and communicate with it using the standard TVM RPC
# protocol. The Model Library Format serves as the model input to this process. When embedded
# platforms provide such an integration, they can be used directly by TVM for both host-driven
# inference and autotuning . This integration is provided by the
# `microTVM Project API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_,

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

# Create a temporary directory

generated_project_dir = gen_path / "generated-project"
if generated_project_dir.is_dir():
    shutil.rmtree(generated_project_dir)
generated_project = tvm.micro.generate_project(template_project_path, module, generated_project_dir, project_options)

# Build and flash the project
generated_project.build()
generated_project.flash()


######################################################################
# Next, establish a session with the simulated device and run the
# computation. The `with session` line would typically flash an attached
# microcontroller, but in this tutorial, it simply launches a subprocess
# to stand in for an attached microcontroller.

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    if args.profile:
        graph_mod = tvm.micro.create_local_debug_executor(
            module.get_graph_json(), session.get_system_lib(), session.device
        )
    else:
        graph_mod = tvm.micro.create_local_graph_executor(
            module.get_graph_json(), session.get_system_lib(), session.device
        )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**module.get_params())

    # graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))

    graph_mod.run()

    tvm_output = graph_mod.get_output(0).numpy()
    print("result is: " + str(tvm_output))
