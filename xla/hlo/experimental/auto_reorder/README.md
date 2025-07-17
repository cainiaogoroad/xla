# Auto Reorder

## run tests
```
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE="auto_reorder=5,auto_reorder_solver=5"
bazel test --compilation_mode=dbg xla/hlo/experimental/auto_reorder:auto_reorder_test --test_filter="AutoReorderingTest.ReorderPassAOpt" --incompatible_strict_action_env --action_env=USE_CUDA --action_env=XLA_CUDA --jobs=8 --flaky_test_attempts=1 --test_output=errors --action_env=CUDA_TOOLKIT_PATH=/usr/local/cuda
```



## run Database PGLE test:

```
export TF_CPP_MIN_LOG_LEVEL=0 
export TF_CPP_VMODULE="auto_reorder=5,auto_reorder_solver=5,gpu_collective_performance_model=5,gpu_performance_model=5,convert_xplane=5"

bazel run --compilation_mode=dbg xla/hlo/experimental/auto_reorder:offline_sqlite_pgle_test 
--incompatible_strict_action_env --action_env=USE_CUDA --action_env=XLA_CUDA --jobs=8 --action_env=CUDA_TOOLKIT_PATH=/usr/local/cuda
```

build auto reorder tools
1. convert xplane to sqlite file
2. convert xplane to chrome_trace_json

## Convert Profiler file to sqlite file
build auto reorder tools, convert xplane to sqlite file

```bash
chmod +x ./convert_xplane_tools
LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/torch/lib ./convert_xplane_tools <xplane_dir> <output_filename>
#example:
./convert_xplane_tools /root/profiler/llama_xla_trace_1n8g_b10/plugins/profile/2024_07_23_15_00_11/ llama_xla_trace_1n8g_b10.db
```
convert_xplane_tools is build by bazel:

```
export HERMETIC_PYTHON_VERSION=3.8
bazel build xla/hlo/experimental/auto_reorder:convert_xplane_tools --incompatible_strict_action_env --action_env=USE_CUDA --action_env=XLA_CUDA --action_env=HERMETIC_PYTHON_VERSION --jobs=64  --config=monolithic --action_env=CUDA_TOOLKIT_PATH=/usr/local/cuda
```
