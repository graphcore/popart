# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# TODO(T42953): Make all tests run on IpuModel once T42732 solved.
add_popart_py_unit_test(testtiedgather VARIANTS IpuModel MATCHEXPR "not test_tied_gather_pattern_correctness")
# `test_tied_gather_pattern_correctness[...]` tests start here
set(splits_params "1" "4")
set(transpose_on_gather_params "True" "False")
set(mean_reduction_strategy_params "mean_reduction_strategy0"
                                   "mean_reduction_strategy1")
set(phase_optimizer_params "fwd-None" "bwd-Sgd" "bwd-Lamb")

foreach(splits ${splits_params})
  foreach(transpose_on_gather ${transpose_on_gather_params})
    foreach(mean_reduction_strategy ${mean_reduction_strategy_params})
      foreach(phase_optimizer ${phase_optimizer_params})
        string(
          JOIN
          "-"
          matchexpr
          "${phase_optimizer}"
          "${mean_reduction_strategy}"
          "${transpose_on_gather}"
          "${splits}")
        set(matchexpr "test_tied_gather_pattern_correctness[${matchexpr}]")
        add_popart_py_unit_test(testtiedgather VARIANTS Hw MATCHEXPR
                                "${matchexpr}")
      endforeach()
    endforeach()
  endforeach()
endforeach()
# `test_tied_gather_pattern_correctness[...]` tests end here
