#!/usr/bin/env bash

/home/user/tensorflow/bazel-bin/tensorflow/core/profiler/profiler help

/home/user/tensorflow/bazel-bin/tensorflow/core/profiler/profiler \
    --profile_path="./profile_dir/profile_499" \
    --op_log_path="./tfprof_log"